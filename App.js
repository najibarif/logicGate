import React, { useEffect, useRef, useState, useCallback } from 'react';
import { StyleSheet, Text, View, TouchableOpacity, Image, Platform, ScrollView, useWindowDimensions, Animated } from 'react-native';
import { Camera } from 'expo-camera';
import * as tf from '@tensorflow/tfjs';

const LABELS = ['OR', 'AND', 'NOT']; // adjust to your model
const IMAGE_SIZE = 224;
const BAR_COLORS = ['#28a745', '#d9534f', '#8a63ff'];

export default function App() {
  const { width: windowWidth } = useWindowDimensions();
  const isNarrow = windowWidth < 760;

  const cameraRef = useRef(null);
  const videoRef = useRef(null);
  const webStreamRef = useRef(null);
  const modelRef = useRef(null);

  const [hasPermission, setHasPermission] = useState(null);
  const [cameraReady, setCameraReady] = useState(false);
  const [videoReady, setVideoReady] = useState(false);
  const [webNeedsStart, setWebNeedsStart] = useState(false);

  const [modelLoading, setModelLoading] = useState(false);
  const [isTfReady, setIsTfReady] = useState(false);

  const [liveRunning, setLiveRunning] = useState(false);
  const liveRef = useRef({ running: false });

  const [predictions, setPredictions] = useState([]);
  const [photoUri, setPhotoUri] = useState(null);

  const nativeFrameIntervalMs = 700;

  // pulsing animation for LIVE badge
  const pulseAnim = useRef(new Animated.Value(1)).current;
  useEffect(() => {
    if (liveRunning) {
      const loop = Animated.loop(
        Animated.sequence([
          Animated.timing(pulseAnim, { toValue: 1.15, duration: 600, useNativeDriver: true }),
          Animated.timing(pulseAnim, { toValue: 1.0, duration: 600, useNativeDriver: true }),
        ])
      );
      loop.start();
      return () => loop.stop();
    } else {
      pulseAnim.setValue(1);
    }
  }, [liveRunning, pulseAnim]);

  // base64 helper (native)
  const atobPoly = useCallback((input) => {
    if (typeof global.atob === 'function') return global.atob(input);
    if (typeof Buffer !== 'undefined') return Buffer.from(input, 'base64').toString('binary');
    throw new Error('atob not available.');
  }, []);
  function base64ToUint8Array(base64) {
    const binaryString = atobPoly(base64);
    const len = binaryString.length;
    const bytes = new Uint8Array(len);
    for (let i = 0; i < len; i++) bytes[i] = binaryString.charCodeAt(i);
    return bytes;
  }

  // permissions
  useEffect(() => {
    (async () => {
      if (Platform.OS === 'web') {
        setHasPermission(true);
        setWebNeedsStart(true);
      } else {
        const { status } = await Camera.requestCameraPermissionsAsync();
        setHasPermission(status === 'granted');
      }
    })();

    return () => {
      // cleanup on unmount
      liveRef.current.running = false;
      setLiveRunning(false);
      stopWebLoop();
      stopNativeLoop();

      if (webStreamRef.current) {
        try { webStreamRef.current.getTracks().forEach(t => t.stop()); } catch (e) { }
        webStreamRef.current = null;
      }
      try { if (modelRef.current && modelRef.current.dispose) modelRef.current.dispose(); } catch (e) { }
      modelRef.current = null;
    };
  }, []);

  // ---------- model loader ----------
  const loadModel = useCallback(async () => {
    if (modelRef.current || modelLoading) return modelRef.current;
    setModelLoading(true);
    try {
      await tf.ready();
      setIsTfReady(true);

      let m = null;
      if (Platform.OS === 'web') {
        const jsonModule = await import('./assets/model/model.json');
        const modelJson = jsonModule.default || jsonModule;
        const weightsUrl = require('./assets/model/weights.bin');
        const weightsResp = await fetch(weightsUrl);
        const weightsData = await weightsResp.arrayBuffer();
        const weightSpecs = Array.isArray(modelJson.weightsManifest)
          ? modelJson.weightsManifest.flatMap((g) => g.weights || [])
          : [];
        const ioHandler = tf.io.fromMemory({
          modelTopology: modelJson.modelTopology,
          weightSpecs,
          weightData: weightsData,
        });
        m = await tf.loadLayersModel(ioHandler);
      } else {
        const { bundleResourceIO } = await import('@tensorflow/tfjs-react-native');
        const modelJson = require('./assets/model/model.json');
        const modelWeights = [require('./assets/model/weights.bin')];
        m = await tf.loadLayersModel(bundleResourceIO(modelJson, modelWeights));
      }

      if (m) modelRef.current = m;
      return m;
    } catch (e) {
      console.error('loadModel error', e);
      alert('Gagal memuat model. Lihat console.');
      return null;
    } finally {
      setModelLoading(false);
    }
  }, [modelLoading]);

  // ---------- loops ----------
  const webLoopRef = useRef({ rafId: null, lastRun: 0, frameIntervalMs: 180 });
  const runWebLoop = useCallback(() => {
    const step = async () => {
      webLoopRef.current.rafId = requestAnimationFrame(step);
      const now = Date.now();
      if (now - webLoopRef.current.lastRun < webLoopRef.current.frameIntervalMs) return;
      webLoopRef.current.lastRun = now;

      if (!videoRef.current || !modelRef.current) return;
      try {
        const probs = tf.tidy(() => {
          const t = tf.browser.fromPixels(videoRef.current);
          const r = tf.image.resizeBilinear(t, [IMAGE_SIZE, IMAGE_SIZE]);
          const norm = r.div(255.0).expandDims(0);
          const out = modelRef.current.predict(norm);
          let data = null;
          if (Array.isArray(out)) {
            if (out[0].dataSync) data = out[0].dataSync();
          } else if (out.dataSync) {
            data = out.dataSync();
          }
          return data ? Array.from(data) : null;
        });

        if (probs) {
          const arr = probs.map((p, i) => ({ className: LABELS[i] || `class_${i}`, probability: p }));
          arr.sort((a, b) => b.probability - a.probability);
          setPredictions(arr);
        }
      } catch (e) {
        console.warn('webLoop predict error', e);
      }
    };
    webLoopRef.current.rafId = requestAnimationFrame(step);
  }, []);

  const stopWebLoop = useCallback(() => {
    if (webLoopRef.current.rafId) {
      cancelAnimationFrame(webLoopRef.current.rafId);
      webLoopRef.current.rafId = null;
    }
  }, []);

  const nativeLoopRef = useRef({ timeoutId: null });
  const runNativeLoop = useCallback(async () => {
    if (!cameraRef.current || !modelRef.current) return;
    try {
      const p = await cameraRef.current.takePictureAsync({ base64: true, quality: 0.4, skipProcessing: true });
      setPhotoUri(p.uri);

      let imageTensor = null;
      try {
        const u8 = base64ToUint8Array(p.base64);
        if (tf.node && tf.node.decodeImage) imageTensor = tf.node.decodeImage(u8, 3);
        else if (tf.decodeJpeg) imageTensor = tf.decodeJpeg(u8);
      } catch (e) {
        console.warn('native decode error', e);
      }
      if (!imageTensor) {
        console.warn('native: cannot decode image to tensor');
      } else {
        const resized = tf.image.resizeBilinear(imageTensor, [IMAGE_SIZE, IMAGE_SIZE]);
        const normalized = resized.div(255.0).expandDims(0);
        const out = tf.tidy(() => {
          const preds = modelRef.current.predict(normalized);
          let data = null;
          if (Array.isArray(preds)) {
            if (preds[0].dataSync) data = preds[0].dataSync();
          } else if (preds.dataSync) {
            data = preds.dataSync();
          }
          return data ? Array.from(data) : null;
        });
        tf.dispose([imageTensor, resized, normalized]);
        if (out) {
          const arr = out.map((p, i) => ({ className: LABELS[i] || `class_${i}`, probability: p }));
          arr.sort((a, b) => b.probability - a.probability);
          setPredictions(arr);
        }
      }
    } catch (e) {
      console.warn('runNativeLoop error', e);
    } finally {
      if (liveRef.current.running) {
        nativeLoopRef.current.timeoutId = setTimeout(runNativeLoop, nativeFrameIntervalMs);
      }
    }
  }, []);

  const stopNativeLoop = useCallback(() => {
    if (nativeLoopRef.current.timeoutId) {
      clearTimeout(nativeLoopRef.current.timeoutId);
      nativeLoopRef.current.timeoutId = null;
    }
  }, []);

  // ---------- start live tied to camera ----------
  const startLiveInternal = useCallback(async () => {
    if (liveRef.current.running) return true;
    let m = modelRef.current;
    if (!m) m = await loadModel();
    if (!m) {
      alert('Gagal memuat model.');
      return false;
    }

    liveRef.current.running = true;
    setLiveRunning(true);

    if (Platform.OS === 'web') {
      webLoopRef.current.frameIntervalMs = 180;
      runWebLoop();
    } else {
      runNativeLoop();
    }
    return true;
  }, [loadModel, runWebLoop, runNativeLoop]);

  // ---------- web: start camera and auto-start live ----------
  const startWebCamera = useCallback(async () => {
    if (Platform.OS !== 'web') return;
    try {
      const constraints = { video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: 'environment' }, audio: false };
      const stream = await navigator.mediaDevices.getUserMedia(constraints);
      webStreamRef.current = stream;
      setHasPermission(true);
      setWebNeedsStart(false);

      if (videoRef.current) {
        videoRef.current.muted = true;
        videoRef.current.playsInline = true;
        videoRef.current.srcObject = stream;
        try {
          await videoRef.current.play();
          setVideoReady(true);
          // after video is playing, auto-start live
          await startLiveInternal();
        } catch (e) {
          setWebNeedsStart(true);
          console.warn('video play failed', e);
        }
      } else {
        await startLiveInternal();
      }
    } catch (err) {
      console.error('startWebCamera error', err);
      setWebNeedsStart(true);
      setHasPermission(false);
      alert('Tidak bisa mengakses kamera. Periksa permission di browser.');
    }
  }, [startLiveInternal]);

  // ---------- native: when camera ready, auto-start live ----------
  const handleCameraReady = useCallback(async () => {
    setCameraReady(true);
    await startLiveInternal();
  }, [startLiveInternal]);

  // ---------- snapshot ----------
  const handleSnapshot = useCallback(async () => {
    if (!modelRef.current) await loadModel();
    if (!modelRef.current) return;

    if (Platform.OS === 'web') {
      if (!videoRef.current) return alert('Video belum siap');
      const v = videoRef.current;
      const w = v.videoWidth || 300;
      const h = v.videoHeight || 300;
      const canvas = document.createElement('canvas'); canvas.width = w; canvas.height = h;
      const ctx = canvas.getContext('2d'); ctx.drawImage(v, 0, 0, w, h);
      setPhotoUri(canvas.toDataURL('image/jpeg', 0.8));

      const probs = tf.tidy(() => {
        const t = tf.browser.fromPixels(v);
        const r = tf.image.resizeBilinear(t, [IMAGE_SIZE, IMAGE_SIZE]);
        const norm = r.div(255.0).expandDims(0);
        const out = modelRef.current.predict(norm);
        let data = null;
        if (Array.isArray(out)) {
          if (out[0].dataSync) data = out[0].dataSync();
        } else if (out.dataSync) {
          data = out.dataSync();
        }
        return data ? Array.from(data) : null;
      });
      if (probs) {
        const arr = probs.map((p, i) => ({ className: LABELS[i] || `class_${i}`, probability: p }));
        arr.sort((a, b) => b.probability - a.probability);
        setPredictions(arr);
      }
    } else {
      if (!cameraRef.current) return;
      const p = await cameraRef.current.takePictureAsync({ base64: true, quality: 0.6, skipProcessing: true });
      setPhotoUri(p.uri);
      let imageTensor = null;
      try {
        const u8 = base64ToUint8Array(p.base64);
        if (tf.node && tf.node.decodeImage) imageTensor = tf.node.decodeImage(u8, 3);
        else if (tf.decodeJpeg) imageTensor = tf.decodeJpeg(u8);
      } catch (e) { console.warn(e); }
      if (imageTensor) {
        const r = tf.image.resizeBilinear(imageTensor, [IMAGE_SIZE, IMAGE_SIZE]);
        const norm = r.div(255.0).expandDims(0);
        const res = tf.tidy(() => {
          const preds = modelRef.current.predict(norm);
          let data = null;
          if (Array.isArray(preds)) {
            if (preds[0].dataSync) data = preds[0].dataSync();
          } else if (preds.dataSync) {
            data = preds.dataSync();
          }
          return data ? Array.from(data) : null;
        });
        tf.dispose([imageTensor, r, norm]);
        if (res) {
          const arr = res.map((p, i) => ({ className: LABELS[i] || `class_${i}`, probability: p }));
          arr.sort((a, b) => b.probability - a.probability);
          setPredictions(arr);
        }
      }
    }
  }, [loadModel]);

  // ---------- UI: bar component ----------
  const ProbBar = ({ label, pct, color, highlight }) => {
    return (
      <View style={[styles.probRow, highlight ? styles.probRowHighlight : null]}>
        <View style={styles.probTextRow}>
          <Text style={[styles.probLabel, highlight ? styles.topLabel : null]}>{label}</Text>
          <Text style={[styles.probPct, highlight ? styles.topLabel : null]}>{pct}%</Text>
        </View>
        <View style={styles.progressBg}>
          <View style={[styles.progressFill, { width: `${pct}%`, backgroundColor: color }]} />
        </View>
      </View>
    );
  };

  // ---------- renderBar list ----------
  const renderBars = () => {
    const top = predictions[0]?.className;
    return LABELS.map((label, idx) => {
      const p = predictions.find((x) => x.className === label) || { probability: 0 };
      const pct = Math.round((p.probability || 0) * 100);
      const barColor = BAR_COLORS[idx] || '#1976d2';
      const highlight = top === label && pct > 0;
      return <ProbBar key={label} label={label} pct={pct} color={barColor} highlight={highlight} />;
    });
  };

  // ---------- render ----------
  if (hasPermission === null) return <View style={styles.center}><Text>Meminta izin kamera...</Text></View>;
  if (hasPermission === false) return <View style={styles.center}><Text>Tidak ada akses kamera</Text></View>;

  const cameraCardStyle = isNarrow ? styles.cardFull : styles.cardCamera;
  const outputCardStyle = isNarrow ? styles.cardFull : styles.cardOutput;

  const topPrediction = predictions[0];

  return (
    <View style={styles.container}>
      <ScrollView contentContainerStyle={styles.scrollContent}>

        <View style={styles.headerRow}>
          <View>
            <Text style={styles.title}>Logic Gate Classifier</Text>
            <Text style={styles.subtitle}>Realtime inference • Teachable Machine model</Text>
          </View>
          <View style={styles.headerRight}>
            {modelLoading ? (
              <View style={styles.modelStatus}><Text style={styles.modelStatusText}>Memuat model…</Text></View>
            ) : modelRef.current ? (
              <View style={styles.modelOk}><Text style={styles.modelOkText}>Model siap</Text></View>
            ) : (
              <View style={styles.modelWarn}><Text style={styles.modelWarnText}>Belum dimuat</Text></View>
            )}
          </View>
        </View>

        <View style={[styles.mainRow, { alignSelf: 'center', maxWidth: 980, width: '100%' }]}>
          {/* Camera Card */}
          <View style={[cameraCardStyle, styles.card]}>
            <View style={styles.cameraHeader}>
              <Text style={styles.cardTitle}>Kamera</Text>
              {liveRunning && (
                <Animated.View style={[styles.liveBadge, { transform: [{ scale: pulseAnim }] }]}>
                  <Text style={styles.liveText}>LIVE</Text>
                </Animated.View>
              )}
            </View>

            <View style={styles.cameraBox}>
              {Platform.OS === 'web' ? (
                <View style={{ flex: 1 }}>
                  <video ref={videoRef} autoPlay playsInline muted style={{ width: '100%', height: '100%', objectFit: 'cover' }} />
                  {webNeedsStart && (
                    <View style={styles.startOverlay}>
                      <TouchableOpacity style={styles.startButton} onPress={startWebCamera}>
                        <Text style={styles.startButtonText}>Mulai Kamera</Text>
                      </TouchableOpacity>
                    </View>
                  )}
                </View>
              ) : (
                <Camera style={{ flex: 1 }} ref={cameraRef} onCameraReady={handleCameraReady} />
              )}
            </View>

            {photoUri ? <Image source={{ uri: photoUri }} style={styles.snapshot} /> : null}
          </View>

          {/* Output Card */}
          <View style={[outputCardStyle, styles.card]}>
            <View style={styles.outputHeader}>
              <Text style={styles.cardTitle}>Output</Text>
              <Text style={styles.smallMuted}>{liveRunning ? 'Inference aktif' : 'Inference mati'}</Text>
            </View>

            <View style={styles.outputContent}>
              {renderBars()}

              {topPrediction ? (
                <View style={styles.topPrediction}>
                  <Text style={styles.topLabelText}>Prediksi teratas</Text>
                  <Text style={styles.topPredictionText}>{topPrediction.className} — {(topPrediction.probability * 100).toFixed(1)}%</Text>
                </View>
              ) : (
                <Text style={styles.smallMuted}>Arahkan kamera ke contoh untuk melihat prediksi</Text>
              )}
            </View>

            <View style={styles.outputActions}>
              <TouchableOpacity style={styles.snapshotButton} onPress={handleSnapshot}>
                <Text style={styles.snapshotButtonText}>📸 Snapshot</Text>
              </TouchableOpacity>
            </View>
          </View>
        </View>

        <View style={{ height: 48 }} />

      </ScrollView>
    </View>
  );
}

// ---------- styles ----------
const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#f6f8fb' },
  scrollContent: { paddingHorizontal: 20, paddingTop: 20, paddingBottom: 40 },

  headerRow: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', marginBottom: 18 },
  title: { fontSize: 20, fontWeight: '800', color: '#0f172a' },
  subtitle: { marginTop: 4, color: '#556070', fontSize: 13 },

  headerRight: { flexDirection: 'row', alignItems: 'center' },
  modelStatus: { paddingHorizontal: 10, paddingVertical: 6, backgroundColor: '#eef2ff', borderRadius: 999 },
  modelStatusText: { color: '#444', fontSize: 12 },
  modelOk: { paddingHorizontal: 10, paddingVertical: 6, backgroundColor: '#e6ffef', borderRadius: 999 },
  modelOkText: { color: '#166534', fontSize: 12 },
  modelWarn: { paddingHorizontal: 10, paddingVertical: 6, backgroundColor: '#fff4e6', borderRadius: 999 },
  modelWarnText: { color: '#92400e', fontSize: 12 },

  mainRow: { flexDirection: 'row', gap: 20, justifyContent: 'center' },

  // card base
  card: {
    backgroundColor: '#fff',
    borderRadius: 14,
    padding: 14,
    // shadows
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 6 },
    shadowOpacity: 0.06,
    shadowRadius: 12,
    elevation: 6,
  },

  // camera card styles
  cardCamera: { flex: 1, marginRight: 12, minWidth: 360, maxWidth: 720 },
  cameraHeader: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', marginBottom: 10 },
  cardTitle: { fontSize: 16, fontWeight: '700', color: '#0f172a' },
  liveBadge: {
    backgroundColor: '#ff3b30',
    paddingHorizontal: 10,
    paddingVertical: 6,
    borderRadius: 999,
    shadowColor: '#ff3b30',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.18,
    shadowRadius: 8,
    elevation: 4,
  },
  liveText: { color: '#fff', fontWeight: '800', fontSize: 12 },

  cameraBox: {
    height: 420,
    borderRadius: 10,
    overflow: 'hidden',
    backgroundColor: '#000',
  },
  startOverlay: { position: 'absolute', inset: 0, alignItems: 'center', justifyContent: 'center', backgroundColor: 'rgba(0,0,0,0.36)' },
  startButton: { paddingHorizontal: 24, paddingVertical: 12, backgroundColor: '#2563eb', borderRadius: 10 },
  startButtonText: { color: '#fff', fontWeight: '700' },

  snapshot: { width: 140, height: 140, position: 'absolute', right: 18, bottom: 18, borderRadius: 8, borderWidth: 2, borderColor: '#fff' },

  // output card styles
  cardOutput: { width: 340, paddingVertical: 16, paddingHorizontal: 14 },
  outputHeader: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12 },
  smallMuted: { color: '#667085', fontSize: 12 },

  outputContent: { gap: 8 },

  // probability row
  probRow: { marginBottom: 8 },
  probRowHighlight: { backgroundColor: '#f8fafb', padding: 8, borderRadius: 8 },
  probTextRow: { flexDirection: 'row', justifyContent: 'space-between', marginBottom: 8 },
  probLabel: { fontWeight: '700', color: '#0f172a' },
  probPct: { fontWeight: '700', color: '#0f172a' },

  progressBg: {
    height: 12,
    backgroundColor: '#eef2ff',
    borderRadius: 999,
    overflow: 'hidden',
  },
  progressFill: {
    height: '100%',
    borderRadius: 999,
  },

  topPrediction: { marginTop: 14, padding: 10, backgroundColor: '#f8fafc', borderRadius: 8, alignItems: 'flex-start' },
  topLabelText: { fontSize: 12, color: '#475569', marginBottom: 6 },
  topPredictionText: { fontSize: 16, fontWeight: '800', color: '#0f172a' },

  outputActions: { marginTop: 12, flexDirection: 'row', justifyContent: 'center', alignItems: 'center'},
  snapshotButton: { backgroundColor: '#0f172a', paddingHorizontal: 14, paddingVertical: 10, borderRadius: 10 },
  snapshotButtonText: { color: '#fff', fontWeight: '700' },

  // responsive fallbacks (stack)
  cardFull: { width: '100%', padding: 14, marginBottom: 18 },

  center: { flex: 1, alignItems: 'center', justifyContent: 'center' },
});

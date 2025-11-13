import React, { useEffect, useRef, useState, useCallback } from 'react';
import { StyleSheet, Text, View, TouchableOpacity, Image, Platform, ScrollView } from 'react-native';
import { Camera } from 'expo-camera';
import * as tf from '@tensorflow/tfjs';

/**
 * Live inference App.js - full fix
 *
 * Perbaikan utama:
 * - modelRef.current digunakan untuk menyimpan instance model (imperative ref)
 * - loadModel() mengembalikan instance model dan langsung menyimpan ke modelRef.current
 * - predictFromTensor selalu memakai modelRef.current
 * - startLive menunggu hasil loadModel() dan memeriksa modelRef.current (bukan hanya state)
 *
 * Sesuaikan LABELS, IMAGE_SIZE, dan jalur model jika perlu.
 */

const LABELS = ['OR', 'AND', 'NOT']; // ganti sesuai model
const IMAGE_SIZE = 224;

// Bar colors: OR changed to green (#28a745)
const BAR_COLORS = ['#28a745', '#d9534f', '#8a63ff'];

export default function App() {
  const cameraRef = useRef(null);
  const videoRef = useRef(null);
  const webStreamRef = useRef(null);

  // imperative ref for model (keuntungan: tersedia sync setelah load)
  const modelRef = useRef(null);

  const [hasPermission, setHasPermission] = useState(null);
  const [cameraReady, setCameraReady] = useState(false);
  const [videoReady, setVideoReady] = useState(false);
  const [webNeedsStart, setWebNeedsStart] = useState(false);

  const [model, setModel] = useState(null); // kept for UI/debug
  const [modelLoading, setModelLoading] = useState(false);
  const [isTfReady, setIsTfReady] = useState(false);

  const [liveRunning, setLiveRunning] = useState(false);
  const liveRef = useRef({ running: false }); // mutable ref to control loop

  const [predictions, setPredictions] = useState([]);
  const [photoUri, setPhotoUri] = useState(null);

  // For native: interval between captures (ms). Increase to reduce CPU.
  const nativeFrameIntervalMs = 700;

  // --------- Helper: base64 polyfill (untuk native decoding) ----------
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

  // ---------- mount: permissions ----------
  useEffect(() => {
    (async () => {
      if (Platform.OS === 'web') {
        setHasPermission(true);
        setWebNeedsStart(true); // require user gesture to start camera on web
      } else {
        const { status } = await Camera.requestCameraPermissionsAsync();
        setHasPermission(status === 'granted');
      }
    })();

    return () => {
      stopLive();
      // cleanup web stream if any
      if (webStreamRef.current) {
        try { webStreamRef.current.getTracks().forEach(t => t.stop()); } catch(e) {}
        webStreamRef.current = null;
      }
      // dispose model if present
      try {
        if (modelRef.current && modelRef.current.dispose) modelRef.current.dispose();
      } catch (e) { /* ignore */ }
      modelRef.current = null;
    };
  }, []);

  // ---------- start web camera on user gesture ----------
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
        try { await videoRef.current.play(); setVideoReady(true); } catch (e) { setWebNeedsStart(true); console.warn('video play failed', e); }
      }
    } catch (err) {
      console.error('startWebCamera error', err);
      setWebNeedsStart(true);
      setHasPermission(false);
      alert('Tidak bisa mengakses kamera. Periksa permission di browser.');
    }
  }, []);

  // ---------- model loader (lazy) ----------
  const loadModel = useCallback(async () => {
    // if already loaded in ref, return it immediately
    if (modelRef.current || modelLoading) return modelRef.current;
    setModelLoading(true);
    try {
      await tf.ready();
      setIsTfReady(true);

      let m = null;
      if (Platform.OS === 'web') {
        // web: try loading bundled model (adjust path if needed)
        // Note: bundlers may not handle binary require; if fetch fails, consider placing model in public/
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
        console.log('Model loaded (web)');
      } else {
        // native (tfjs-react-native)
        const { bundleResourceIO } = await import('@tensorflow/tfjs-react-native');
        const modelJson = require('./assets/model/model.json');
        const modelWeights = [require('./assets/model/weights.bin')];
        m = await tf.loadLayersModel(bundleResourceIO(modelJson, modelWeights));
        console.log('Model loaded (native)');
      }

      if (m) {
        modelRef.current = m; // set ref synchronously
        setModel(m);          // update state for UI/debug
      }
      return m;
    } catch (e) {
      console.error('loadModel error', e);
      alert('Gagal memuat model. Lihat console.');
      return null;
    } finally {
      setModelLoading(false);
    }
  }, [modelLoading]);

  // ---------- inference helpers ----------
  const predictFromTensor = useCallback((tensor) => {
    const m = modelRef.current;
    if (!m) return null;
    // tensor shape [1, IMAGE_SIZE, IMAGE_SIZE, 3]
    return tf.tidy(() => {
      const out = m.predict(tensor);
      let data = null;
      if (Array.isArray(out)) {
        if (out[0].dataSync) data = out[0].dataSync();
      } else if (out.dataSync) {
        data = out.dataSync();
      }
      if (!data) return null;
      const arr = Array.from(data).map((p, i) => ({ className: LABELS[i] || `class_${i}`, probability: p }));
      arr.sort((a, b) => b.probability - a.probability);
      return arr;
    });
  }, []);

  // ---------- web live loop using requestAnimationFrame ----------
  const webLoopRef = useRef({ rafId: null, lastRun: 0, frameIntervalMs: 200 }); // default 5 FPS
  const runWebLoop = useCallback(() => {
    const step = async (time) => {
      webLoopRef.current.rafId = requestAnimationFrame(step);
      const now = Date.now();
      if (now - webLoopRef.current.lastRun < webLoopRef.current.frameIntervalMs) return;
      webLoopRef.current.lastRun = now;

      if (!videoRef.current || !modelRef.current) return;
      try {
        const results = await tf.tidy(() => {
          const t = tf.browser.fromPixels(videoRef.current);
          const resized = tf.image.resizeBilinear(t, [IMAGE_SIZE, IMAGE_SIZE]);
          const normalized = resized.div(255.0).expandDims(0);
          // predictFromTensor uses modelRef
          const out = modelRef.current.predict(normalized);
          let data = null;
          if (Array.isArray(out)) {
            if (out[0].dataSync) data = out[0].dataSync();
          } else if (out.dataSync) {
            data = out.dataSync();
          }
          // return plain JS array so tidy won't keep tensors
          return data ? Array.from(data) : null;
        });

        if (results) {
          const arr = results.map((p, i) => ({ className: LABELS[i] || `class_${i}`, probability: p }));
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

  // ---------- native live loop using periodic takePictureAsync ----------
  const nativeLoopRef = useRef({ timeoutId: null });
  const runNativeLoop = useCallback(async () => {
    if (!cameraRef.current || !modelRef.current) return;
    try {
      const p = await cameraRef.current.takePictureAsync({ base64: true, quality: 0.4, skipProcessing: true });
      setPhotoUri(p.uri);
      // decode base64 to tensor
      let imageTensor = null;
      try {
        const u8 = base64ToUint8Array(p.base64);
        if (tf.node && tf.node.decodeImage) {
          imageTensor = tf.node.decodeImage(u8, 3);
        } else if (tf.decodeJpeg) {
          imageTensor = tf.decodeJpeg(u8);
        }
      } catch (e) {
        console.warn('native decode error', e);
      }
      if (!imageTensor) {
        console.warn('native: cannot decode image to tensor');
      } else {
        const resized = tf.image.resizeBilinear(imageTensor, [IMAGE_SIZE, IMAGE_SIZE]);
        const normalized = resized.div(255.0).expandDims(0);
        const results = predictFromTensor(normalized); // uses modelRef
        tf.dispose([imageTensor, resized, normalized]);
        if (results) setPredictions(results);
      }
    } catch (e) {
      console.warn('runNativeLoop error', e);
    } finally {
      // schedule next run
      if (liveRef.current.running) {
        nativeLoopRef.current.timeoutId = setTimeout(runNativeLoop, nativeFrameIntervalMs);
      }
    }
  }, [predictFromTensor]);

  const stopNativeLoop = useCallback(() => {
    if (nativeLoopRef.current.timeoutId) {
      clearTimeout(nativeLoopRef.current.timeoutId);
      nativeLoopRef.current.timeoutId = null;
    }
  }, []);

  // ---------- start/stop live ----------
  const startLive = useCallback(async () => {
    if (liveRef.current.running) return;
    // ensure camera available
    if (Platform.OS === 'web') {
      if (!videoRef.current || !videoRef.current.srcObject) {
        await startWebCamera();
      }
      if (!videoRef.current || !videoRef.current.srcObject) {
        alert('Kamera belum aktif. Tekan Mulai Kamera terlebih dahulu.');
        return;
      }
    } else {
      if (!cameraRef.current) {
        alert('Kamera native belum siap.');
        return;
      }
    }

    // pastikan model ada: gunakan modelRef atau loadModel() yang mengembalikan model
    let m = modelRef.current;
    if (!m) {
      m = await loadModel();
    }
    if (!m) {
      alert('Gagal memuat model.');
      return;
    }

    liveRef.current.running = true;
    setLiveRunning(true);

    if (Platform.OS === 'web') {
      webLoopRef.current.frameIntervalMs = 200; // ~5 fps
      runWebLoop();
    } else {
      runNativeLoop();
    }
  }, [loadModel, runWebLoop, runNativeLoop, startWebCamera]);

  const stopLive = useCallback(() => {
    liveRef.current.running = false;
    setLiveRunning(false);
    stopWebLoop();
    stopNativeLoop();
  }, [stopWebLoop, stopNativeLoop]);

  // ---------- UI helper: render probability bars ----------
  const renderBars = () => {
    return LABELS.map((label, idx) => {
      const p = predictions.find((x) => x.className === label) || { probability: 0 };
      const pct = Math.round((p.probability || 0) * 100);
      const barColor = BAR_COLORS[idx] || '#1976d2';
      return (
        <View key={label} style={{ marginVertical: 6 }}>
          <Text style={{ fontWeight: '600' }}>{label}</Text>
          <View style={{ height: 22, backgroundColor: '#f1f1f1', borderRadius: 6, overflow: 'hidden', marginTop: 6 }}>
            <View style={{ width: `${pct}%`, height: '100%', backgroundColor: barColor, justifyContent: 'center' }}>
              <Text style={{ color: '#fff', paddingLeft: 8, fontWeight: '700' }}>{pct > 5 ? `${pct}%` : ''}</Text>
            </View>
          </View>
        </View>
      );
    });
  };

  // ---------- render ----------
  if (hasPermission === null) return <View style={styles.center}><Text>Meminta izin kamera...</Text></View>;
  if (hasPermission === false) return <View style={styles.center}><Text>Tidak ada akses kamera</Text></View>;

  return (
    <View style={styles.container}>
      <ScrollView contentContainerStyle={styles.scrollContent}>
        <Text style={styles.header}>Live Logic Gate Classifier</Text>

        {/* Row: Kamera | Output */}
        <View style={styles.rowContainer}>
          <View style={styles.cameraWrapper}>
            {Platform.OS === 'web' ? (
              <View style={[styles.camera, { position: 'relative' }]}>
                <video ref={videoRef} autoPlay playsInline muted style={{ width: '100%', height: '100%', objectFit: 'cover' }} />
                {webNeedsStart && (
                  <View style={styles.startOverlay}>
                    <TouchableOpacity style={styles.startButton} onPress={startWebCamera}>
                      <Text style={styles.startButtonText}>Mulai Kamera</Text>
                    </TouchableOpacity>
                  </View>
                )}
                {!videoReady && !webNeedsStart && (
                  <View style={styles.badge}><Text style={{ color: '#fff' }}>Menunggu kamera...</Text></View>
                )}
              </View>
            ) : (
              <Camera style={styles.camera} ref={cameraRef} onCameraReady={() => setCameraReady(true)} />
            )}
          </View>

          <View style={styles.outputWrapper}>
            <Text style={{ fontWeight: 'bold', marginBottom: 6 }}>Output</Text>
            {renderBars()}
            {photoUri && <Image source={{ uri: photoUri }} style={styles.preview} />}
            {modelLoading && <Text style={{ marginTop: 8 }}>Memuat model...</Text>}
            {modelRef.current && !modelLoading && <Text style={{ marginTop: 8 }}>Model siap</Text>}
          </View>
        </View>

        <View style={{ flexDirection: 'row', marginTop: 12 }}>
          <TouchableOpacity
            style={[styles.button, liveRunning ? { backgroundColor: '#d9534f' } : {}]}
            onPress={() => (liveRunning ? stopLive() : startLive())}
          >
            <Text style={styles.buttonText}>{liveRunning ? 'Hentikan Live' : 'Mulai Live'}</Text>
          </TouchableOpacity>

          <TouchableOpacity
            style={[styles.button, { backgroundColor: '#6c757d' }]}
            onPress={() => {
              // snapshot & single classify (if user wants)
              (async () => {
                // ensure model is loaded (use loadModel which sets modelRef)
                if (!modelRef.current) await loadModel();
                if (!modelRef.current) return;

                if (Platform.OS === 'web') {
                  if (!videoRef.current) return alert('Video belum siap');
                  const v = videoRef.current;
                  const w = v.videoWidth || 300; const h = v.videoHeight || 300;
                  const canvas = document.createElement('canvas'); canvas.width = w; canvas.height = h;
                  const ctx = canvas.getContext('2d'); ctx.drawImage(v, 0, 0, w, h);
                  setPhotoUri(canvas.toDataURL('image/jpeg', 0.8));

                  // use tf.tidy and modelRef directly to avoid depending on state
                  const probs = await tf.tidy(() => {
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
                    const res = predictFromTensor(norm);
                    tf.dispose([imageTensor, r, norm]);
                    if (res) setPredictions(res);
                  }
                }
              })();
            }}
          >
            <Text style={styles.buttonText}>Snapshot</Text>
          </TouchableOpacity>
        </View>

        <View style={{ height: 40 }} />
      </ScrollView>
    </View>
  );
}

// ---------- styles ----------
const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#fff', padding: 12 },
  header: { fontSize: 18, fontWeight: 'bold', marginBottom: 8, textAlign: 'center' },
  scrollContent: { paddingBottom: 40 },
  // Row container to put camera and output side-by-side
  rowContainer: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    justifyContent: 'space-between',
    gap: 12, // RN may ignore gap; margins used on children
  },
  cameraWrapper: { flex: 1, marginRight: 8, minWidth: 200, maxWidth: 520 },
  outputWrapper: { width: 300, paddingLeft: 4 },
  camera: { width: '100%', height: 360, borderRadius: 8, overflow: 'hidden', backgroundColor: '#000' },
  startOverlay: { position: 'absolute', inset: 0, alignItems: 'center', justifyContent: 'center', backgroundColor: 'rgba(0,0,0,0.25)' },
  startButton: { paddingHorizontal: 24, paddingVertical: 12, backgroundColor: '#1976d2', borderRadius: 10 },
  startButtonText: { color: '#fff', fontWeight: '700' },
  badge: { position: 'absolute', top: 8, left: 8, backgroundColor: '#0008', padding: 6, borderRadius: 6 },
  button: { backgroundColor: '#1976d2', padding: 12, borderRadius: 8, marginTop: 12, marginRight: 8 },
  buttonText: { color: '#fff', textAlign: 'center', fontWeight: '600' },
  preview: { width: 120, height: 120, marginTop: 12, alignSelf: 'flex-start' },
  center: { flex: 1, alignItems: 'center', justifyContent: 'center' },
});

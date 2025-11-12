import React, { useEffect, useRef, useState } from 'react';
import { StyleSheet, Text, View, TouchableOpacity, Image, Platform } from 'react-native';
import { Camera, CameraView } from 'expo-camera';
import * as FileSystem from 'expo-file-system';
import * as ImagePicker from 'expo-image-picker';

import * as tf from '@tensorflow/tfjs';

export default function App() {
  const cameraRef = useRef(null);
  const videoRef = useRef(null);
  const [hasPermission, setHasPermission] = useState(null);
  const [isTfReady, setIsTfReady] = useState(false);
  const [model, setModel] = useState(null);
  const [predictions, setPredictions] = useState([]);
  const [photoUri, setPhotoUri] = useState(null);
  const [loading, setLoading] = useState(false);
  const webStreamRef = useRef(null);
  const [webNeedsStart, setWebNeedsStart] = useState(false);

  // Label & image size sesuai metadata.json yang Anda upload
  const labels = ['OR', 'AND']; // sesuai metadata.json
  const IMAGE_SIZE = 224; // sesuai metadata.json

  useEffect(() => {
    (async () => {
      if (Platform.OS === 'web') {
        await initWebStream();
      } else {
        const { status } = await Camera.requestCameraPermissionsAsync();
        setHasPermission(status === 'granted');
      }

      // inisialisasi tfjs backend
      await tf.ready();
      setIsTfReady(true);

      try {
        if (Platform.OS === 'web') {
          // Web: dynamically import JSON and fetch weights asset, then load from memory
          const jsonModule = await import('./assets/model/model.json');
          const modelJson = jsonModule.default || jsonModule;
          const weightsUrl = require('./assets/model/weights.bin');
          const weightsResp = await fetch(weightsUrl);
          const weightsData = await weightsResp.arrayBuffer();
          // Flatten weight specs from manifest groups
          const weightSpecs = Array.isArray(modelJson.weightsManifest)
            ? modelJson.weightsManifest.flatMap((g) => g.weights || [])
            : [];
          // Use object form to avoid deprecation warning
          const ioHandler = tf.io.fromMemory({
            modelTopology: modelJson.modelTopology,
            weightSpecs,
            weightData: weightsData,
          });
          const loadedModel = await tf.loadLayersModel(ioHandler);
          setModel(loadedModel);
          console.log('Model loaded (web)');
        } else {
          // Native: use tfjs-react-native dynamically with bundleResourceIO
          const { bundleResourceIO } = await import('@tensorflow/tfjs-react-native');
          const modelJson = require('./assets/model/model.json');
          const modelWeights = [require('./assets/model/weights.bin')];
          const loadedModel = await tf.loadLayersModel(bundleResourceIO(modelJson, modelWeights));
          setModel(loadedModel);
          console.log('Model loaded (native)');
        }
      } catch (err) {
        console.warn('Gagal memuat model. Pastikan model.json dan weights.bin ada di assets/model dan filename cocok.');
        console.error(err);
      }
    })();

    // cleanup web stream on unmount
    return () => {
      if (webStreamRef.current) {
        webStreamRef.current.getTracks().forEach((t) => t.stop());
        webStreamRef.current = null;
      }
    };
  }, []);

  const initWebStream = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment' }, audio: false });
      setHasPermission(true);
      webStreamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await new Promise((resolve) => {
          const v = videoRef.current;
          const handler = () => {
            v.removeEventListener('loadedmetadata', handler);
            resolve();
          };
          v.addEventListener('loadedmetadata', handler);
        });
        try { await videoRef.current.play(); } catch {}
      }
      setWebNeedsStart(false);
    } catch (e) {
      // jika user gesture diperlukan atau izin ditolak, tampilkan tombol manual
      console.warn('Web camera init failed:', e && e.message ? e.message : e);
      setHasPermission(true); // tetap render UI; sediakan tombol manual
      setWebNeedsStart(true);
    }
  };

  // polyfill atob jika belum tersedia
  const atobPoly = (input) => {
    if (typeof global.atob === 'function') return global.atob(input);
    // Buffer tersedia di banyak environment; jika tidak ada, ini bisa gagal
    if (typeof Buffer !== 'undefined') return Buffer.from(input, 'base64').toString('binary');
    throw new Error('atob not available. Install a base64 polyfill or run in environment with atob/Buffer.');
  };

  function base64ToUint8Array(base64) {
    const binaryString = atobPoly(base64);
    const len = binaryString.length;
    const bytes = new Uint8Array(len);
    for (let i = 0; i < len; i++) {
      bytes[i] = binaryString.charCodeAt(i);
    }
    return bytes;
  }

  const takeAndClassify = async () => {
    if (!model) return;
    setLoading(true);
    try {
      let photo = null;
      if (Platform.OS === 'web') {
        // Web: ambil frame dari video stream jika ada, jika tidak fallback ke file picker
        if (videoRef.current && videoRef.current.readyState >= 2) {
          const videoEl = videoRef.current;
          const w = videoEl.videoWidth || 300;
          const h = videoEl.videoHeight || 300;
          const canvas = document.createElement('canvas');
          canvas.width = w; canvas.height = h;
          const ctx = canvas.getContext('2d');
          ctx.drawImage(videoEl, 0, 0, w, h);
          const dataUrl = canvas.toDataURL('image/jpeg', 0.8);
          photo = { uri: dataUrl };
          setPhotoUri(dataUrl);
        } else {
          const file = await new Promise((resolve, reject) => {
            const input = document.createElement('input');
            input.type = 'file';
            input.accept = 'image/*';
            input.setAttribute('capture', 'environment');
            input.onchange = () => resolve(input.files && input.files[0]);
            input.onerror = (e) => reject(e);
            input.click();
          });
          if (!file) throw new Error('No file selected');
          const objectUrl = URL.createObjectURL(file);
          photo = { uri: objectUrl };
          setPhotoUri(objectUrl);
        }
      } else {
        // Native: Ambil foto (base64 supaya bisa didecode)
        if (!cameraRef.current) { setLoading(false); return; }
        photo = await cameraRef.current.takePictureAsync({
          base64: true,
          quality: 0.6,
          skipProcessing: true,
        });
        setPhotoUri(photo.uri);
      }

      // Buat tensor gambar sesuai platform
      let imageTensor = null;
      if (Platform.OS === 'web') {
        try {
          if (videoRef.current && videoRef.current.readyState >= 2) {
            imageTensor = tf.browser.fromPixels(videoRef.current);
          } else {
            const img = new window.Image();
            img.crossOrigin = 'anonymous';
            img.src = photo.uri;
            await new Promise((res, rej) => {
              img.onload = () => res();
              img.onerror = (err) => rej(err);
            });
            imageTensor = tf.browser.fromPixels(img);
          }
        } catch (e) {
          console.warn('Web decode failed:', e.message);
        }
      } else {
        try {
          // Native: coba decode dari base64 (Uint8Array) menggunakan tfjs decodeJpeg / node.decodeImage
          const u8 = base64ToUint8Array(photo.base64);
          if (tf.node && tf.node.decodeImage) {
            imageTensor = tf.node.decodeImage(u8, 3);
          } else if (tf.decodeJpeg) {
            imageTensor = tf.decodeJpeg(u8);
          }
        } catch (e) {
          console.warn('Gagal mendecode langsung via tfjs decodeJpeg/decodeImage, error:', e.message);
        }

        if (!imageTensor) {
          try {
            const b64 = await FileSystem.readAsStringAsync(photo.uri, { encoding: FileSystem.EncodingType.Base64 });
            const u8b = base64ToUint8Array(b64);
            if (tf.decodeJpeg) {
              imageTensor = tf.decodeJpeg(u8b);
            } else if (tf.node && tf.node.decodeImage) {
              imageTensor = tf.node.decodeImage(u8b, 3);
            }
          } catch (e) {
            console.warn('Fallback decode failed:', e.message);
          }
        }
      }

      if (!imageTensor) {
        // jika tetap gagal, beri tahu user
        setPredictions([{ className: 'error', probability: 0 }]);
        setLoading(false);
        return;
      }

      // resize & normalisasi sesuai model Teachable Machine (0..1)
      const resized = tf.image.resizeBilinear(imageTensor, [IMAGE_SIZE, IMAGE_SIZE]);
      const normalized = resized.div(255.0).expandDims(0); // shape [1, IMAGE_SIZE, IMAGE_SIZE, 3]

      // prediksi
      const out = model.predict(normalized);

      // kemungkinan output: Tensor or array. Ambil data
      let data = null;
      if (Array.isArray(out)) {
        // kalau model mengembalikan array (misalnya [probs])
        if (out[0].dataSync) data = out[0].dataSync();
      } else if (out.dataSync) {
        data = out.dataSync();
      }

      if (data) {
        const results = Array.from(data).map((p, i) => ({ className: labels[i] || `class_${i}`, probability: p }));
        results.sort((a, b) => b.probability - a.probability);
        setPredictions(results);
      } else {
        setPredictions([{ className: 'unknown', probability: 0 }]);
      }

      // cleanup
      tf.dispose([imageTensor, resized, normalized, out]);
    } catch (err) {
      console.error('takeAndClassify error:', err);
    }
    setLoading(false);
  };

  // --- RENDER ---
  if (hasPermission === null) return <View style={styles.center}><Text>Meminta izin kamera...</Text></View>;
  if (hasPermission === false) return <View style={styles.center}><Text>Tidak ada akses kamera</Text></View>;

  return (
    <View style={styles.container}>
      <Text style={styles.header}>Logic Gate Classifier (Teachable Machine)</Text>
      <View style={styles.cameraContainer}>
        {Platform.OS === 'web' ? (
          <View style={[styles.camera, { borderWidth: 1, borderColor: '#ccc', position: 'relative' }]}>
            {/* eslint-disable-next-line jsx-a11y/media-has-caption */}
            <video
              ref={videoRef}
              autoPlay
              playsInline
              muted
              style={{ width: '100%', height: '100%', objectFit: 'cover' }}
            />
            {webNeedsStart && (
              <TouchableOpacity
                onPress={initWebStream}
                style={{ position: 'absolute', bottom: 8, right: 8, backgroundColor: '#0008', paddingHorizontal: 10, paddingVertical: 6, borderRadius: 6 }}
              >
                <Text style={{ color: '#fff' }}>Aktifkan Kamera</Text>
              </TouchableOpacity>
            )}
          </View>
        ) : (
          <CameraView style={styles.camera} ref={cameraRef} />
        )}
      </View>

      <TouchableOpacity style={styles.button} onPress={takeAndClassify} disabled={!model || loading}>
        <Text style={styles.buttonText}>{loading ? 'Memproses...' : 'Ambil Foto & Klasifikasi'}</Text>
      </TouchableOpacity>

      {photoUri && <Image source={{ uri: photoUri }} style={styles.preview} />}

      <View style={styles.predContainer}>
        <Text style={{ fontWeight: 'bold' }}>Prediksi:</Text>
        {isTfReady ? (model ? (
          predictions.length ? (
            predictions.map((p, idx) => (
              <Text key={idx}>{p.className} — {(p.probability * 100).toFixed(1)}%</Text>
            ))
          ) : (
            <Text>- Tekan tombol untuk mulai -</Text>
          )
        ) : (
          <Text>Model belum dimuat (periksa assets/model)</Text>
        )) : (
          <Text>Mempersiapkan TensorFlow...</Text>
        )}
      </View>

      <View style={{ height: 40 }} />
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#fff', padding: 12 },
  header: { fontSize: 18, fontWeight: 'bold', marginBottom: 8, textAlign: 'center' },
  cameraContainer: { alignItems: 'center', justifyContent: 'center' },
  camera: { width: 300, height: 300, borderRadius: 8, overflow: 'hidden' },
  button: { backgroundColor: '#1976d2', padding: 12, borderRadius: 8, marginTop: 12 },
  buttonText: { color: '#fff', textAlign: 'center', fontWeight: '600' },
  preview: { width: 120, height: 120, marginTop: 12, alignSelf: 'center' },
  predContainer: { marginTop: 12 },
  center: { flex: 1, alignItems: 'center', justifyContent: 'center' },
});

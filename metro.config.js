// Learn more https://docs.expo.io/guides/customizing-metro
const { getDefaultConfig } = require('expo/metro-config');

/** @type {import('expo/metro-config').MetroConfig} */
const config = getDefaultConfig(__dirname);

// Ensure TensorFlow weight files (.bin) are treated as assets so `require()` works
// and bundleResourceIO can load them at runtime.
config.resolver = config.resolver || {};
config.resolver.assetExts = config.resolver.assetExts || [];
if (!config.resolver.assetExts.includes('bin')) {
  config.resolver.assetExts.push('bin');
}

module.exports = config;

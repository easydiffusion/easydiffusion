import { defineConfig } from "vite";
import eslint from "vite-plugin-eslint";
import react from "@vitejs/plugin-react";
import { vanillaExtractPlugin } from "@vanilla-extract/vite-plugin";

import path from "path";
// https://vitejs.dev/config/
export default defineConfig({
  resolve: {
    alias: {
      // TODO figure out why vs code complains about this even though it works
      "@api": path.resolve(__dirname, "./src/api"),
      "@stores": path.resolve(__dirname, "./src/stores"),
      "@components": path.resolve(__dirname, "./src/components"),
      "@recipes": path.resolve(__dirname, "./src/components/_recipes"),
      "@atoms": path.resolve(__dirname, "./src/components/atoms"),
      "@molecules": path.resolve(__dirname, "./src/components/molecules"),
      "@organisms": path.resolve(__dirname, "./src/components/organisms"),
      "@layouts": path.resolve(__dirname, "./src/components/layouts"),
      "@pages": path.resolve(__dirname, "./src/pages"),
      "@styles": path.resolve(__dirname, "./src/styles"),
      "@translations": path.resolve(__dirname, "./src/Translation"),
    },
  },

  plugins: [
    eslint(),
    react(),
    vanillaExtractPlugin({
      // configuration
    }),
  ],

  server: {
    port: 9001,
  },

  build: {
    // make sure everythign is in the same directory
    outDir: "../dist",
    rollupOptions: {
      output: {
        // dont hash the file names
        // maybe once we update the python server?
        entryFileNames: `[name].js`,
        chunkFileNames: `[name].js`,
        assetFileNames: `[name].[ext]`,
      },
    },
  },
});

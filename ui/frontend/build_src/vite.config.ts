import { defineConfig } from "vite";
import eslint from 'vite-plugin-eslint'
import react from "@vitejs/plugin-react";
import { vanillaExtractPlugin } from "@vanilla-extract/vite-plugin";

import path from "path";
// https://vitejs.dev/config/
export default defineConfig({
  resolve: {
    alias: {
      // TODO figure out why vs code complains about this even though it works
      "@stores": path.resolve(__dirname, "./src/stores"),
      // TODO - add more aliases
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

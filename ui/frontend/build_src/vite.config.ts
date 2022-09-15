import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { vanillaExtractPlugin } from '@vanilla-extract/vite-plugin';


// https://vitejs.dev/config/
export default defineConfig({
  plugins: [
    react(),
    vanillaExtractPlugin({
      // configuration
    })
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

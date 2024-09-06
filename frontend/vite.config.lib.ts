import { fileURLToPath, URL } from "node:url";

import vue from "@vitejs/plugin-vue";
import autoprefixer from "autoprefixer";
import { resolve } from "path";
import postcssNesting from "postcss-nesting";
import { defineConfig } from "vite";
import vueDevTools from "vite-plugin-vue-devtools";

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [vue(), vueDevTools()],
  resolve: {
    alias: {
      "@": fileURLToPath(new URL("./src", import.meta.url)),
    },
  },
  css: {
    postcss: {
      plugins: [autoprefixer, postcssNesting],
    },
  },
  define: {
    "process.env": {},
  },
  build: {
    lib: {
      entry: resolve(__dirname, "src/share.ts"),
      name: "skore",
      fileName: "skore",
    },
  },
});

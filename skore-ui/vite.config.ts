import { fileURLToPath, URL } from "node:url";

import vue from "@vitejs/plugin-vue";
import autoprefixer from "autoprefixer";
import postcssNesting from "postcss-nesting";
import { defineConfig } from "vite";
import vueDevTools from "vite-plugin-vue-devtools";

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [
    vue({
      template: {
        compilerOptions: {
          isCustomElement: (tag) => ["perspective-viewer"].includes(tag),
        },
      },
    }),
    vueDevTools(),
  ],
  resolve: {
    alias: {
      "@": fileURLToPath(new URL("./src", import.meta.url)),
      "@finos/perspective": "@finos/perspective/dist/esm/perspective.inline.js",
      "@finos/perspective-viewer":
        "@finos/perspective-viewer/dist/esm/perspective-viewer.inline.js",
      "@finos/perspective-styles": "@finos/perspective-viewer/dist/css",
    },
  },
  css: {
    postcss: {
      plugins: [autoprefixer, postcssNesting],
    },
  },
  test: {
    setupFiles: ["./vitest.setup.ts"],
  },
  optimizeDeps: {
    esbuildOptions: {
      target: "es2022",
    },
  },
  build: {
    target: "es2022",
  },
});

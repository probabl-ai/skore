import { readFileSync } from "node:fs";
import { fileURLToPath, URL } from "node:url";

import vue from "@vitejs/plugin-vue";
import autoprefixer from "autoprefixer";
import postcssNesting from "postcss-nesting";
import { defineConfig, Plugin } from "vite";
import vueDevTools from "vite-plugin-vue-devtools";

const base64Loader: Plugin = {
  name: "base64-loader",
  transform(_: any, id: string) {
    const [path, query] = id.split("?");
    if (query != "base64") return null;

    const data = readFileSync(path);
    const base64 = data.toString("base64");

    return `export default '${base64}';`;
  },
};

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [vue(), vueDevTools(), base64Loader],
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
  test: {
    setupFiles: ["./vitest.setup.ts"],
  },
  build: {
    assetsInlineLimit(filePath) {
      const fontExtensions = ["ttf", "woff", "woff2", "svg", "eot"];
      return fontExtensions.some((ext) => filePath.includes(ext));
    },
    rollupOptions: {
      output: {
        inlineDynamicImports: true,
        entryFileNames: `assets/[name].js`,
        chunkFileNames: `assets/[name].js`,
        assetFileNames: `assets/[name].[ext]`,
      },
    },
  },
});

import "@/assets/styles/main.css";
import "simplebar-vue/dist/simplebar.min.css";

import { createPinia } from "pinia";
import { createApp } from "vue";

import App from "@/ShareApp.vue";
import { DataStore } from "@/models";
import { useCanvasStore } from "@/stores/canvas";

export default function share() {
  const app = createApp(App);
  app.use(createPinia());
  app.mount("#app");

  const m = JSON.parse(document.getElementById("mandr-data")?.innerText || "{}");
  const ds = new DataStore(m.uri, m.payload, m.layout);
  const canvasStore = useCanvasStore();
  canvasStore.setDataStore(ds);
}

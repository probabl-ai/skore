import "@/assets/styles/main.css";
import "simplebar-vue/dist/simplebar.min.css";

import { createPinia } from "pinia";
import { createApp } from "vue";

import App from "@/ShareApp.vue";
import { DataStore } from "@/models";
import { useReportsStore } from "@/stores/reports";

export default function share() {
  const app = createApp(App);
  app.use(createPinia());
  app.mount("#app");

  const m = JSON.parse(document.getElementById("skore-data")?.innerText || "{}");
  const ds = new DataStore(m.uri, m.payload, m.layout);
  const reportsStore = useReportsStore();
  reportsStore.setSelectedReportIfDifferent(ds);
}

import "@/assets/styles/main.css";
import "simplebar-vue/dist/simplebar.min.css";

import { createPinia } from "pinia";
import { createApp } from "vue";

import App from "@/ShareApp.vue";
import { useReportStore } from "@/stores/report";

export default function share() {
  const app = createApp(App);
  app.use(createPinia());
  app.mount("#app");

  const m = JSON.parse(document.getElementById("skore-data")?.innerText || "{}");
  const reportsStore = useReportStore();
  reportsStore.setReport(m);
}

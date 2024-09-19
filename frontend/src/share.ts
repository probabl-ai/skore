import "@/assets/styles/main.css";
import "simplebar-vue/dist/simplebar.min.css";

import { createPinia } from "pinia";
import { createApp } from "vue";

import App from "@/ShareApp.vue";
import { useReportStore } from "@/stores/report";

export default async function share() {
  const app = createApp(App);
  app.use(createPinia());
  app.mount("#app");

  const m = JSON.parse(document.getElementById("project-data")?.innerText || "{}");
  const reportsStore = useReportStore();
  await reportsStore.setReport(m);
}

import "@/assets/styles/main.css";
import "simplebar-vue/dist/simplebar.min.css";

import { createPinia } from "pinia";
import { createApp } from "vue";

import App from "@/ShareApp.vue";
import { useProjectStore } from "@/stores/project";

export default async function share(selectedView: string) {
  const app = createApp(App);
  app.use(createPinia());
  app.mount("#app");

  const m = JSON.parse(document.getElementById("project-data")?.innerText || "{}");
  const projectStore = useProjectStore();
  await projectStore.setProject(m);
  projectStore.setCurrentView(selectedView);
}

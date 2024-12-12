import { createPinia } from "pinia";
import { createApp } from "vue";

import "@/assets/styles/main.css";

import router from "@/router";
import SkoreUi from "@/SkoreUi.vue";
import StandaloneWidget from "@/StandaloneWidget.vue";

const pinia = createPinia();
const appContainer = document.getElementById("skore-ui");
if (appContainer !== null) {
  const app = createApp(SkoreUi);
  app.use(pinia);
  app.use(router);
  app.mount(appContainer);
} else {
  const containerId = `skore-widget-${window.skoreWidgetId}`;
  const standaloneWidgetContainer = document.getElementById(containerId);
  if (standaloneWidgetContainer) {
    const app = createApp(StandaloneWidget);
    app.use(pinia);
    app.mount(standaloneWidgetContainer);
  }
}

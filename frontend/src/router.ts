import { createRouter, createWebHashHistory } from "vue-router";

import ComponentsView from "./views/ComponentsView.vue";
import ReportBuilderView from "./views/ReportBuilderView.vue";

export enum ROUTE_NAMES {
  REPORT_BUILDER = "report-builder",
  COMPONENTS = "components",
}

const router = createRouter({
  history: createWebHashHistory(import.meta.env.BASE_URL),
  routes: [
    {
      path: "/:segments*",
      name: "report-builder",
      component: ReportBuilderView,
    },
    {
      path: "/components",
      name: "components",
      component: ComponentsView,
    },
  ],
});

export default router;

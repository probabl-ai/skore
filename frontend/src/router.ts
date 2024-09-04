import { createRouter, createWebHashHistory } from "vue-router";

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
      component: () => import("./views/ComponentsView.vue"),
    },
  ],
});

export default router;

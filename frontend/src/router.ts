import { createRouter, createWebHashHistory } from "vue-router";

import DashboardView from "./views/DashboardView.vue";

const router = createRouter({
  history: createWebHashHistory(import.meta.env.BASE_URL),
  routes: [
    {
      path: "/:segments*",
      name: "dashboard",
      component: DashboardView,
    },
    {
      path: "/components",
      name: "components",
      component: () => import("./views/ComponentsView.vue"),
    },
  ],
});

export default router;

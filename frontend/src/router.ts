import { createRouter, createWebHashHistory } from "vue-router";

import ComponentsView from "./views/ComponentsView.vue";
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
      component: ComponentsView,
    },
  ],
});

export default router;

import { createRouter, createWebHashHistory } from "vue-router";

import ProjectView from "./views/project/ProjectView.vue";

export enum ROUTE_NAMES {
  VIEW_BUILDER = "view-builder",
  COMPONENTS = "components",
}

const router = createRouter({
  history: createWebHashHistory(import.meta.env.BASE_URL),
  routes: [
    {
      path: "/",
      name: "view-builder",
      component: ProjectView,
      meta: {
        icon: "icon-pie-chart",
      },
    },
    {
      path: "/components",
      name: "components",
      component: () => import("./views/ComponentsView.vue"),
      meta: {
        icon: "icon-gift",
      },
    },
  ],
});

export default router;

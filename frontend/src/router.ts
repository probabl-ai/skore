import { createRouter, createWebHashHistory } from "vue-router";

import ProjectView from "./views/ProjectView.vue";

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
    },
    {
      path: "/components",
      name: "components",
      component: () => import("./views/ComponentsView.vue"),
    },
  ],
});

export default router;

import { createRouter, createWebHashHistory, type RouteRecordRaw } from "vue-router";

import ProjectView from "./views/project/ProjectView.vue";

export enum ROUTE_NAMES {
  VIEW_BUILDER = "view-builder",
  COMPONENTS = "components",
}

const routes: RouteRecordRaw[] = [
  {
    path: "/",
    name: "view-builder",
    component: ProjectView,
    meta: {
      icon: "icon-pie-chart",
    },
  },
];

if (import.meta.env.DEV) {
  routes.push({
    path: "/components",
    name: "components",
    component: () => import("./views/ComponentsView.vue"),
    meta: {
      icon: "icon-gift",
    },
  });
}

const router = createRouter({
  history: createWebHashHistory(import.meta.env.BASE_URL),
  routes,
});

export default router;

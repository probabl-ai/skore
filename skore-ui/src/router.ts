import { createRouter, createWebHashHistory, type RouteRecordRaw } from "vue-router";

import ActivityFeedView from "@/views/activity/ActivityFeedView.vue";
import ProjectView from "@/views/project/ProjectView.vue";

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
  {
    path: "/activity",
    name: "activity-feed",
    component: ActivityFeedView,
    meta: {
      icon: "icon-list-sparkle",
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

import { createRouter, createWebHashHistory, type RouteRecordRaw } from "vue-router";

import ActivityFeedView from "@/views/activity/ActivityFeedView.vue";
import ProjectView from "@/views/project/ProjectView.vue";

export enum ROUTE_NAMES {
  VIEW_BUILDER = "view-builder",
  ACTIVITY_FEED = "activity-feed",
  COMPONENTS = "components",
}

const routes: RouteRecordRaw[] = [
  {
    path: "/",
    name: ROUTE_NAMES.ACTIVITY_FEED,
    component: ActivityFeedView,
    meta: {
      icon: "icon-list-sparkle",
    },
  },
  {
    path: "/views",
    name: ROUTE_NAMES.VIEW_BUILDER,
    component: ProjectView,
    meta: {
      icon: "icon-pie-chart",
    },
  },
];

if (import.meta.env.DEV) {
  routes.push({
    path: "/components",
    name: ROUTE_NAMES.COMPONENTS,
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

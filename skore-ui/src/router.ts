import { createRouter as cr, createWebHashHistory, type RouteRecordRaw } from "vue-router";

import ActivityFeedView from "@/views/activity/ActivityFeedView.vue";

export enum ROUTE_NAMES {
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

export function createRouter() {
  return cr({
    history: createWebHashHistory(import.meta.env.BASE_URL),
    routes,
  });
}

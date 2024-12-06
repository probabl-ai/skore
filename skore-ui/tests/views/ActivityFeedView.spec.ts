import { VueWrapper } from "@vue/test-utils";
import { createPinia, setActivePinia } from "pinia";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { createApp } from "vue";
import { useRoute } from "vue-router";

import { ROUTE_NAMES } from "@/router";
import ActivityFeedView from "@/views/activity/ActivityFeedView.vue";
import { mountSuspense } from "../test.utils";

describe("ActivityFeedView", () => {
  beforeEach(() => {
    vi.mock("vue-router");

    const app = createApp({});
    const pinia = createPinia();
    app.use(pinia);
    setActivePinia(pinia);
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("Renders properly", async () => {
    vi.mocked(useRoute).mockImplementationOnce(() => ({
      fullPath: `/${ROUTE_NAMES.ACTIVITY_FEED}`,
      path: `/${ROUTE_NAMES.ACTIVITY_FEED}`,
      query: {},
      params: {},
      matched: [],
      name: ROUTE_NAMES.ACTIVITY_FEED,
      hash: "",
      redirectedFrom: undefined,
      meta: {},
    }));

    const builder = await mountSuspense(ActivityFeedView);
    // i.e. not a `VueError`
    expect(builder).toBeInstanceOf(VueWrapper);
  });
});

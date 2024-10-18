import { VueWrapper } from "@vue/test-utils";
import { createPinia, setActivePinia } from "pinia";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { createApp } from "vue";
import { useRoute } from "vue-router";

import { ROUTE_NAMES } from "@/router";
import ProjectView from "@/views/project/ProjectView.vue";
import { mountSuspense } from "../test.utils";

vi.mock("@/services/api", () => {
  const fetchProject = vi.fn().mockImplementation(() => {
    return { items: {}, views: [] };
  });
  return { fetchProject };
});

vi.hoisted(() => {
  // required because plotly depends on URL.createObjectURL
  const mockObjectURL = vi.fn();
  window.URL.createObjectURL = mockObjectURL;
  window.URL.revokeObjectURL = mockObjectURL;
});

describe("ProjectView", () => {
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
      fullPath: `/${ROUTE_NAMES.VIEW_BUILDER}`,
      path: `/${ROUTE_NAMES.VIEW_BUILDER}`,
      query: {},
      params: {},
      matched: [],
      name: ROUTE_NAMES.VIEW_BUILDER,
      hash: "",
      redirectedFrom: undefined,
      meta: {},
    }));

    const builder = await mountSuspense(ProjectView);
    // i.e. not a `VueError`
    expect(builder).toBeInstanceOf(VueWrapper);
  });
});

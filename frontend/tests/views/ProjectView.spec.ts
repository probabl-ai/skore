import { VueWrapper } from "@vue/test-utils";
import { createPinia, setActivePinia } from "pinia";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { createApp } from "vue";
import { useRoute } from "vue-router";

import { ROUTE_NAMES } from "@/router";
import ReportBuilderView from "@/views/ProjectView.vue";
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

describe("ReportBuilderView", () => {
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
      fullPath: `/${ROUTE_NAMES.REPORT_BUILDER}`,
      path: `/${ROUTE_NAMES.REPORT_BUILDER}`,
      query: {},
      params: {},
      matched: [],
      name: ROUTE_NAMES.REPORT_BUILDER,
      hash: "",
      redirectedFrom: undefined,
      meta: {},
    }));

    const builder = await mountSuspense(ReportBuilderView);
    // i.e. not a `VueError`
    expect(builder).toBeInstanceOf(VueWrapper);
  });
});

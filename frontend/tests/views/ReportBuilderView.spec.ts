import { VueWrapper } from "@vue/test-utils";
import { createPinia, setActivePinia } from "pinia";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { createApp } from "vue";
import { useRoute } from "vue-router";

import type { ItemType } from "@/models";
import { ROUTE_NAMES } from "@/router";
import ReportBuilderView from "@/views/ReportBuilderView.vue";
import { makeDataStore, mountSuspense } from "../test.utils";

const uri = "/a/b";
const keys: ItemType[] = ["boolean", "integer", "vega", "matplotlib_figure"];

vi.mock("@/services/api", () => {
  const fetchAllManderUris = vi.fn().mockImplementation(() => {
    return [uri];
  });
  const fetchMander = vi.fn().mockImplementation(() => {
    return makeDataStore(uri, keys);
  });
  const putLayout = vi.fn().mockImplementation(() => {
    return makeDataStore(uri, keys);
  });
  return { fetchAllManderUris, fetchMander, putLayout };
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
      fullPath: `/${ROUTE_NAMES.REPORT_BUILDER}/a/b`,
      path: `/${ROUTE_NAMES.REPORT_BUILDER}/a/b`,
      query: {},
      params: {
        segments: ["a", "b"],
      },
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

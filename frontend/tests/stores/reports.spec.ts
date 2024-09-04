import { type ItemType, type KeyLayoutSize } from "@/models";
import { fetchAllManderUris, fetchMander } from "@/services/api";
import { useReportsStore } from "@/stores/reports";
import { createTestingPinia } from "@pinia/testing";
import { setActivePinia } from "pinia";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { makeDataStore } from "../test.utils";

const uri = "/test/fixture";
const infoKeys: ItemType[] = [
  "boolean",
  "integer",
  "number",
  "string",
  "any",
  "array",
  "date",
  "datetime",
  "html",
  "markdown",
  "dataframe",
  "image",
  "cv_results",
  "numpy_array",
  "sklearn_model",
];
const plotKeys: ItemType[] = ["vega", "matplotlib_figure"];

vi.mock("@/services/api", () => {
  const fetchAllManderUris = vi.fn().mockImplementation(() => {
    return [uri];
  });
  const fetchMander = vi.fn().mockImplementation(() => {
    return makeDataStore(uri, [...infoKeys, ...plotKeys]);
  });
  const putLayout = vi.fn().mockImplementation(() => {
    return makeDataStore(uri, [...infoKeys, ...plotKeys]);
  });
  return { fetchAllManderUris, fetchMander, putLayout };
});

describe("Reports store", () => {
  beforeEach(() => {
    setActivePinia(createTestingPinia({ stubActions: false, createSpy: vi.fn }));
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("Can create an empty layout when setting a DataStore with no layout.", () => {
    const ds = makeDataStore(uri, [...infoKeys, ...plotKeys]);
    const reportsStore = useReportsStore();

    reportsStore.setSelectedReportIfDifferent(ds);
    expect(reportsStore.selectedReport?.uri).toEqual(uri);
    expect(reportsStore.layout).toHaveLength(0);
  });

  it("Can create layout item for existing key.", () => {
    const layoutItem = { key: "boolean", size: "large" as KeyLayoutSize };
    const ds = makeDataStore(uri, [...infoKeys, ...plotKeys], [layoutItem]);
    const reportsStore = useReportsStore();

    reportsStore.setSelectedReportIfDifferent(ds);
    expect(reportsStore.layout).toContainEqual(layoutItem);
  });

  it("Can add some layout to an existing key.", () => {
    const ds = makeDataStore(uri, [...infoKeys, ...plotKeys]);
    const reportsStore = useReportsStore();

    reportsStore.setSelectedReportIfDifferent(ds);
    reportsStore.displayKey("boolean");
    reportsStore.setKeyLayoutSize("boolean", "small");
    expect(reportsStore.layout).toHaveLength(1);
    expect(reportsStore.layout).toEqual([{ key: "boolean", size: "small" }]);
    reportsStore.setKeyLayoutSize("unknown", "small");
    expect(reportsStore.layout).toHaveLength(1);
    reportsStore.hideKey("boolean");
    expect(reportsStore.layout).toHaveLength(0);
  });

  it("Can poll the backend.", async () => {
    const reportsStore = useReportsStore();

    reportsStore.selectedReportUri = uri;
    await reportsStore.startBackendPolling();
    expect(fetchAllManderUris).toBeCalled();
    expect(fetchMander).toBeCalled();
  });
});

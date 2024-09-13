import type { ReportItem } from "@/models";
import { fetchReport } from "@/services/api";
import { useReportStore } from "@/stores/report";
import { createTestingPinia } from "@pinia/testing";
import { setActivePinia } from "pinia";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("@/services/api", () => {
  const fetchReport = vi.fn().mockImplementation(() => {});
  return { fetchReport };
});

function makeFakeReport() {
  const epoch = new Date("1970-01-01T00:00:00Z").toISOString();
  const i1 = {
    media_type: "text/markdown",
    value: "",
    updated_at: epoch,
    created_at: epoch,
  } as ReportItem;
  const i2 = {
    media_type: "text/markdown",
    value: "",
    updated_at: epoch,
    created_at: epoch,
  } as ReportItem;
  return {
    layout: [],
    items: {
      Any: i1,
      Array: i2,
    },
  };
}
describe("Reports store", () => {
  beforeEach(() => {
    setActivePinia(createTestingPinia({ stubActions: false, createSpy: vi.fn }));
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("Can create an empty layout when setting a DataStore with no layout.", async () => {
    const reportStore = useReportStore();

    await reportStore.setReport(makeFakeReport());
    expect(reportStore.layout).toHaveLength(0);
  });

  it("Can poll the backend.", async () => {
    const reportStore = useReportStore();

    await reportStore.startBackendPolling();
    expect(fetchReport).toBeCalled();
    reportStore.stopBackendPolling();
  });

  it("Can move keys in layout.", async () => {
    const reportStore = useReportStore();

    await reportStore.setReport(makeFakeReport());
    reportStore.displayKey("Any");
    reportStore.displayKey("Array");
    reportStore.setKeyLayoutSize("Any", "large");
    reportStore.setKeyLayoutSize("Array", "large");
    expect(reportStore.layout).toEqual([
      { key: "Any", size: "large" },
      { key: "Array", size: "large" },
    ]);
    reportStore.moveKey("Array", "up");
    expect(reportStore.layout).toEqual([
      { key: "Array", size: "large" },
      { key: "Any", size: "large" },
    ]);
    reportStore.moveKey("Array", "down");
    expect(reportStore.layout).toEqual([
      { key: "Any", size: "large" },
      { key: "Array", size: "large" },
    ]);
  });
});

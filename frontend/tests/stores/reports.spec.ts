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

  it("Can transform keys to a tree", async () => {
    const reportStore = useReportStore();

    const epoch = new Date("1970-01-01T00:00:00Z").toISOString();
    function makeFakeReportItem() {
      return {
        media_type: "text/markdown",
        value: "",
        updated_at: epoch,
        created_at: epoch,
      } as ReportItem;
    }
    const report = {
      layout: [],
      items: {
        a: makeFakeReportItem(),
        "a/b": makeFakeReportItem(),
        "a/b/d": makeFakeReportItem(),
        "a/b/e": makeFakeReportItem(),
        "a/b/f/g": makeFakeReportItem(),
      },
    };
    await reportStore.setReport(report);
    expect(reportStore.keysAsTree()).toEqual([
      {
        name: "a",
        children: [
          { name: "a (self)", children: [] },
          {
            name: "a/b",
            children: [
              { name: "a/b (self)", children: [] },
              { name: "a/b/d", children: [] },
              { name: "a/b/e", children: [] },
              {
                name: "a/b/f",
                children: [
                  { name: "a/b/f/g", children: [] },
                ],
              },
            ],
          },
        ],
      },
    ]);
  });
});

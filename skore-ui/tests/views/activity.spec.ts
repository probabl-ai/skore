import { setActivePinia } from "pinia";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import type { ProjectItemDto } from "@/dto";
import { fetchActivityFeed } from "@/services/api";
import { useActivityStore } from "@/views/activity/activity";
import { createTestingPinia } from "@pinia/testing";

const epoch = new Date("1970-01-01T00:00:00Z").toISOString();
function makeFakeViewItem(name: string, note: string = "") {
  return {
    name,
    media_type: "text/markdown",
    value: "",
    updated_at: epoch,
    created_at: epoch,
    note,
    version: 0,
  } as ProjectItemDto;
}

vi.mock("@/services/api", () => {
  const noop = vi.fn().mockImplementation(() => {});
  return {
    fetchActivityFeed: vi.fn(() => {
      return [makeFakeViewItem("a", ""), makeFakeViewItem("b", "")];
    }),
    setNote: noop,
  };
});

describe("Project store", () => {
  beforeEach(() => {
    setActivePinia(createTestingPinia({ stubActions: false, createSpy: vi.fn }));
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("Can poll the backend.", async () => {
    const activityStore = useActivityStore();

    await activityStore.startBackendPolling();
    expect(fetchActivityFeed).toBeCalled();
    activityStore.stopBackendPolling();
  });

  it("Can add a note on an item", async () => {
    const activityStore = useActivityStore();

    await activityStore.startBackendPolling();
    expect(fetchActivityFeed).toBeCalled();
    activityStore.stopBackendPolling();

    await activityStore.setNoteOnItem("a", 0, "test");
    expect(activityStore.items[0].note).toBe("test");
  });
});

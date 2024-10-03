import { createTestingPinia } from "@pinia/testing";
import { setActivePinia } from "pinia";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { deleteView, fetchProject, fetchShareableBlob, putView } from "@/services/api";
import { useToastsStore } from "@/stores/toasts";
import { createFetchResponse, mockedFetch } from "../test.utils";

describe("API Service", () => {
  beforeEach(() => {
    setActivePinia(createTestingPinia({ stubActions: false, createSpy: vi.fn }));
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("Can report errors.", async () => {
    const error = new Error("Something went wrong");
    mockedFetch.mockImplementation(() => {
      throw error;
    });

    expect(await fetchProject()).toBeNull();
    const toastsStore = useToastsStore();
    expect(toastsStore.toasts.length).toBe(1);
  });

  it("Can call endpoints", async () => {
    mockedFetch.mockResolvedValue(createFetchResponse({}, 200));
    const project = await fetchProject();
    expect(project).toBeDefined();

    mockedFetch.mockResolvedValue(createFetchResponse({}, 201));
    const view = await putView("test", []);
    expect(view).toBeDefined();

    mockedFetch.mockResolvedValue(createFetchResponse({}, 202));
    const del = await deleteView("test");
    expect(del).toBeUndefined();

    mockedFetch.mockResolvedValue(createFetchResponse({}, 200));
    const share = await fetchShareableBlob("test");
    expect(share).toBeDefined();
  });
});

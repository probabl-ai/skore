import type { ProjectItem } from "@/models";
import { fetchProject } from "@/services/api";
import { useProjectStore } from "@/stores/project";
import { createTestingPinia } from "@pinia/testing";
import { setActivePinia } from "pinia";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("@/services/api", () => {
  const fetchProject = vi.fn().mockImplementation(() => {});
  return { fetchProject };
});

describe("Project store", () => {
  beforeEach(() => {
    setActivePinia(createTestingPinia({ stubActions: false, createSpy: vi.fn }));
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("Can poll the backend.", async () => {
    const projectStore = useProjectStore();

    await projectStore.startBackendPolling();
    expect(fetchProject).toBeCalled();
    projectStore.stopBackendPolling();
  });

  it("Can transform keys to a tree", async () => {
    const projectStore = useProjectStore();

    const epoch = new Date("1970-01-01T00:00:00Z").toISOString();
    function makeFakeViewItem() {
      return {
        media_type: "text/markdown",
        value: "",
        updated_at: epoch,
        created_at: epoch,
      } as ProjectItem;
    }
    const project = {
      items: {
        a: makeFakeViewItem(),
        "a/b": makeFakeViewItem(),
        "a/b/d": makeFakeViewItem(),
        "a/b/e": makeFakeViewItem(),
        "a/b/f/g": makeFakeViewItem(),
      },
      views: {},
    };
    await projectStore.setProject(project);
    expect(projectStore.keysAsTree()).toEqual([
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
                children: [{ name: "a/b/f/g", children: [] }],
              },
            ],
          },
        ],
      },
    ]);
  });
});

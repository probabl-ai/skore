import type { ProjectDto, ProjectItemDto } from "@/dto";
import { fetchProject } from "@/services/api";
import { useProjectStore } from "@/stores/project";
import { createTestingPinia } from "@pinia/testing";
import { setActivePinia } from "pinia";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const epoch = new Date("1970-01-01T00:00:00Z").toISOString();
function makeFakeViewItem(name: string, note: string = "") {
  return {
    name,
    media_type: "text/markdown",
    value: "",
    updated_at: epoch,
    created_at: epoch,
    note,
  } as ProjectItemDto;
}

vi.mock("@/services/api", () => {
  const noop = vi.fn().mockImplementation(() => {});
  return { fetchProject: noop, putView: noop, deleteView: noop };
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

    const project = {
      items: {
        a: [makeFakeViewItem("a")],
        "a/b": [makeFakeViewItem("a/b")],
        "a/b/d": [makeFakeViewItem("a/b/d")],
        "a/b/e": [makeFakeViewItem("a/b/e")],
        "a/b/f/g": [makeFakeViewItem("a/b/f/g")],
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

  it("Can get the history of an item", async () => {
    const projectStore = useProjectStore();

    const h1 = makeFakeViewItem("a");
    const h2 = makeFakeViewItem("a");
    const h3 = makeFakeViewItem("a");
    const project: ProjectDto = {
      items: {
        a: [h1, h2, h3],
      },
      views: { default: [{ kind: "item", value: "a" }] },
    };
    await projectStore.setProject(project);

    let d = projectStore.currentViewItems[0];
    expect(d.createdAt.toISOString()).toEqual(h1.created_at);
    expect(d.updatedAt.toISOString()).toEqual(h1.updated_at);
    expect(d.data).toEqual(h1.value);
    expect(d.name).toEqual("a");

    projectStore.setCurrentItemUpdateIndex("a", 1);
    d = projectStore.currentViewItems[0];
    expect(d.createdAt.toISOString()).toEqual(h2.created_at);
    expect(d.updatedAt.toISOString()).toEqual(h2.updated_at);
    expect(d.data).toEqual(h2.value);
    expect(d.name).toEqual("a");

    projectStore.setCurrentItemUpdateIndex("a", 2);
    d = projectStore.currentViewItems[0];
    expect(d.createdAt.toISOString()).toEqual(h3.created_at);
    expect(d.updatedAt.toISOString()).toEqual(h3.updated_at);
    expect(d.data).toEqual(h3.value);
    expect(d.name).toEqual("a");
  });
});

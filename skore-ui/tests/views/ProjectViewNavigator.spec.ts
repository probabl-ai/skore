import { config, shallowMount } from "@vue/test-utils";
import { createPinia, setActivePinia } from "pinia";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { createApp } from "vue";

import { useProjectStore } from "@/stores/project";
import ProjectViewNavigator from "@/views/project/ProjectViewNavigator.vue";

vi.mock("@/services/api", () => {
  const fetchProject = vi.fn(() => {
    return { items: {}, views: {} };
  });
  const putView = vi.fn();
  return { fetchProject, putView };
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

    // Simplebar will be stub but we want it's content to be rendered
    config.global.renderStubDefaultSlot = true;
  });

  afterEach(() => {
    vi.restoreAllMocks();

    config.global.renderStubDefaultSlot = false;
  });

  it("Can name next view with an incremented name", async () => {
    const projectStore = useProjectStore();
    projectStore.createView("New view");

    const wrapper = shallowMount(ProjectViewNavigator, {
      global: {
        stubs: {
          EditableList: false,
          EditableListItem: false,
        },
      },
    });
    const dropdown = wrapper.find(".dropdown");
    expect(dropdown).toBeDefined();
    await dropdown.trigger("click");

    const addViewButton = wrapper.find(".new-view");
    await addViewButton.trigger("click");

    const items = wrapper.findAll(".editable-list-item");
    expect(items.length).toEqual(2);

    const newItem = wrapper.get(".editable-list-item:last-child");
    expect(newItem.text()).toContain("New view 1");
  });

  it("Can increment new view even if list has holes?", async () => {
    const projectStore = useProjectStore();
    projectStore.createView("New view");
    projectStore.createView("New view 2");
    projectStore.createView("New view 5");
    projectStore.createView("New view 6");

    const wrapper = shallowMount(ProjectViewNavigator, {
      global: {
        stubs: {
          EditableList: false,
          EditableListItem: false,
        },
      },
    });
    const dropdown = wrapper.find(".dropdown");
    expect(dropdown).toBeDefined();
    await dropdown.trigger("click");

    const addViewButton = wrapper.find(".new-view");
    await addViewButton.trigger("click");

    const newItem = wrapper.get(".editable-list-item:last-child");
    expect(newItem.text()).toContain("New view 7");
  });
});

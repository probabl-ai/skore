import { config, shallowMount } from "@vue/test-utils";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import DataFrameWidget from "@/components/DataFrameWidget.vue";

function generateFakeProps(numItems: number) {
  const index = [];
  const columns = ["id", "first_name", "last_name", "email", "gender", "ip_address"];
  const data = [];

  for (let i = 1; i <= numItems; i++) {
    index.push([i, `Index${i}`]);
    data.push([
      i,
      `FirstName${i}`,
      `LastName${i}`,
      `user${i}@example.com`,
      i % 2 === 0 ? "female" : "male",
      `192.168.1.${i}`,
    ]);
  }

  return { index, columns, data };
}

function mountComponent(props: any) {
  return shallowMount(DataFrameWidget, {
    props,
    global: {
      stubs: {
        Simplebar: {
          template: "<div><slot /></div>",
        },
      },
    },
  });
}

describe("DataFrameWidget", () => {
  beforeEach(() => {
    config.global.renderStubDefaultSlot = true;
  });

  afterEach(() => {
    config.global.renderStubDefaultSlot = false;
    vi.restoreAllMocks();
  });

  it("Renders properly.", () => {
    const props = generateFakeProps(10);
    const wrapper = mountComponent(props);
    expect(wrapper.findAll("thead > tr > th").length).toBe(props.columns.length + 1);
    expect(wrapper.findAll("tbody > tr").length).toBe(props.data.length);
  });

  it("Computes rows correctly.", () => {
    const props = generateFakeProps(500);
    const wrapper = mountComponent(props);

    const component = wrapper.vm as any;
    expect(component.totalPages).toBe(50);
    expect(component.visibleRows.length).toBe(props.data.length / component.totalPages);
    expect(component.pageStart).toBe(0);
    expect(component.pageEnd).toBe(component.visibleRows.length);

    component.rowPerPage = 50;
    expect(component.totalPages).toBe(10);
    expect(component.visibleRows.length).toBe(50);
  });

  it("Can navigate through pages.", async () => {
    const props = generateFakeProps(500);
    const wrapper = mountComponent(props);

    const [firstPageButton, previousButton, nextButton, lastPageButton] = wrapper.findAll(
      ".pagination-buttons > button"
    );
    const component = wrapper.vm as any;

    expect(firstPageButton.attributes("disabled")).toBeDefined();
    expect(previousButton.attributes("disabled")).toBeDefined();
    expect(nextButton.attributes("disabled")).toBeUndefined();
    expect(lastPageButton.attributes("disabled")).toBeUndefined();

    await nextButton.trigger("click");
    expect(component.currentPage).toBe(1);
    expect(firstPageButton.attributes("disabled")).toBeUndefined();
    expect(previousButton.attributes("disabled")).toBeUndefined();
    expect(nextButton.attributes("disabled")).toBeUndefined();
    expect(lastPageButton.attributes("disabled")).toBeUndefined();
  });

  it("Can search through rows.", () => {
    const props = generateFakeProps(5);
    const wrapper = mountComponent(props);
    const component = wrapper.vm as any;
    component.search = "FirstName1";
    expect(component.visibleRows.length).toBe(1);
  });
});

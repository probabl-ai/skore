import { createTestingPinia } from "@pinia/testing";
import { mount } from "@vue/test-utils";
import { setActivePinia } from "pinia";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { nextTick } from "vue";

import ToastNotificationArea from "@/components/ToastNotificationArea.vue";
import { useToastsStore } from "@/stores/toasts";

describe("ToastNotificationArea", () => {
  beforeEach(() => {
    setActivePinia(createTestingPinia({ stubActions: false, createSpy: vi.fn }));
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("Can render.", () => {
    const wrapper = mount(ToastNotificationArea);
    expect(wrapper.html().length).toBeGreaterThan(0);
  });

  it("Can render all kind of toasts", async () => {
    const toastsStore = useToastsStore();
    toastsStore.addToast("Hello", "info");
    toastsStore.addToast("Hello", "warning");
    toastsStore.addToast("Hello", "error");
    toastsStore.addToast("Hello", "success");

    const wrapper = mount(ToastNotificationArea);
    expect(wrapper.findAll(".toast").length).toEqual(4);
  });

  it("Can show and hide toasts.", async () => {
    const toastsStore = useToastsStore();
    const wrapper = mount(ToastNotificationArea);

    toastsStore.addToast("Hello", "info");
    await nextTick();
    expect(wrapper.html()).toContain("Hello");

    toastsStore.dismissToast(toastsStore.toasts[0].id);
    await nextTick();
    expect(wrapper.html()).not.toContain("Hello");
  });

  it("Can hide manually dismissed toasts.", async () => {
    const toastsStore = useToastsStore();
    const wrapper = mount(ToastNotificationArea);

    toastsStore.addToast("Hello", "info");
    await nextTick();
    expect(wrapper.html()).toContain("Hello");

    await wrapper.find(".actions > button").trigger("click");
    expect(wrapper.html()).not.toContain("Hello");
  });
});

import { useToastsStore } from "@/stores/toasts";
import { createTestingPinia } from "@pinia/testing";
import { setActivePinia } from "pinia";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

describe("Toasts store", () => {
  beforeEach(() => {
    setActivePinia(createTestingPinia({ stubActions: false, createSpy: vi.fn }));
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("Can add toast that have a unique id", async () => {
    const toastsStore = useToastsStore();

    toastsStore.addToast("Hello", "info");
    expect(toastsStore.toasts.length).toBe(1);
    const toast1 = toastsStore.toasts[0];
    expect(toast1.id).toBeDefined();
    expect(toast1.message).toBe("Hello");
    expect(toast1.type).toBe("info");

    toastsStore.addToast("World", "success");
    expect(toastsStore.toasts.length).toBe(2);
    const toast2 = toastsStore.toasts[1];
    expect(toast2.id).toBeDefined();
    expect(toast2.message).toBe("World");
    expect(toast2.type).toBe("success");

    expect(toast1.id).not.toBe(toast2.id);
  });

  it("Can dismiss toast", async () => {
    const toastsStore = useToastsStore();

    toastsStore.addToast("Hello", "info");
    toastsStore.dismissToast(toastsStore.toasts[0].id);
    expect(toastsStore.toasts.length).toBe(0);
  });

  it("Can only add the same message once", () => {
    const toastsStore = useToastsStore();

    toastsStore.addToast("Hello", "info");
    toastsStore.addToast("Hello", "info");

    const toast = toastsStore.toasts[0];
    expect(toastsStore.toasts.length).toBe(1);
    expect(toast.message).toBe("Hello");
    expect(toast.count).toBe(2);

    toastsStore.addToast("Hello", "info");
    expect(toastsStore.toasts.length).toBe(1);
    expect(toast.count).toBe(3);
  });
});

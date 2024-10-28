import { generateRandomId } from "@/services/utils";
import { defineStore } from "pinia";
import { ref } from "vue";

export type ToastType = "info" | "success" | "warning" | "error";

export interface Toast {
  id: string;
  message: string;
  type: ToastType;
  count?: number;
  dismissible?: boolean;
  duration?: number;
}

export const useToastsStore = defineStore("toasts", () => {
  const toasts = ref<Toast[]>([]);

  /**
   * Add a toast to the toasts array.
   * If the message is already in the array, increment the count.
   * If the message is not in the array, add it.
   * @param message The message to add.
   * @param type The type of toast to add.
   * @param options The options for the toas
   * @param options.dismissible Whether the toast can be dismissed by the user. Default is true.
   * @param options.duration The duration in seconds for which the toast should be displayed. Default is Infinity.
   */
  function addToast(
    message: string,
    type: ToastType,
    { dismissible = true, duration = Infinity } = {}
  ) {
    // Check if the message is already in the toasts array
    const existingToast = toasts.value.find(
      (toast) => toast.message === message && toast.type === type
    );
    if (existingToast) {
      // If the message is already in the array, increment the count
      existingToast.count = (existingToast.count || 1) + 1;
    } else {
      // If the message is not in the array, add it
      toasts.value.push({ id: generateRandomId(), message, type, count: 1, dismissible, duration });
    }
  }

  /**
   * Dismiss a toast by its id.
   * @param id The id of the toast to dismiss.
   */
  function dismissToast(id: string) {
    toasts.value = toasts.value.filter((toast) => toast.id !== id);
  }

  return { toasts, addToast, dismissToast };
});

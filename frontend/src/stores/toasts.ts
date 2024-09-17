import { generateRandomId } from "@/services/utils";
import { defineStore } from "pinia";
import { ref } from "vue";

export type ToastType = "info" | "success" | "warning" | "error";

export interface Toast {
  id: string;
  message: string;
  type: ToastType;
}

export const useToastsStore = defineStore("toasts", () => {
  const toasts = ref<Toast[]>([]);

  function addToast(message: string, type: ToastType) {
    toasts.value.push({ id: generateRandomId(), message, type });
  }

  function dismissToast(id: string) {
    toasts.value = toasts.value.filter((toast) => toast.id !== id);
  }

  return { toasts, addToast, dismissToast };
});

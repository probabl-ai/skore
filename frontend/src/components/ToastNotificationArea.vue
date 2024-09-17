<script setup lang="ts">
import ToastNotification from "@/components/ToastNotification.vue";
import { useToastsStore } from "@/stores/toasts";

const toastsStore = useToastsStore();
</script>

<template>
  <div class="toast-notification-area">
    <TransitionGroup name="toasts" tag="div">
      <div v-for="toast in toastsStore.toasts" :key="toast.id">
        <ToastNotification :id="toast.id" :message="toast.message" :type="toast.type" />
      </div>
    </TransitionGroup>
  </div>
</template>

<style scoped>
.toast-notification-area {
  position: fixed;
  z-index: 9000;
  right: 0;
  bottom: 0;
  display: flex;
  width: 60dvw;
  flex-direction: column;
  padding: var(--spacing-padding-large);
  gap: var(--spacing-gap-small);
}

.toasts-move,
.toasts-enter-active,
.toasts-leave-active {
  z-index: 1;
  transition: all 0.5s ease;
}

.toasts-enter-from {
  opacity: 0;
  transform: translateY(30px);
}

.toasts-leave-to {
  opacity: 0;
  transform: translateY(-30px);
}

.toasts-leave-active {
  position: absolute;
  z-index: 2;
}
</style>

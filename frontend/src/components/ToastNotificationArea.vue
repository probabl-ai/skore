<script setup lang="ts">
import ToastNotification from "@/components/ToastNotification.vue";
import { useToastsStore } from "@/stores/toasts";

const toastsStore = useToastsStore();

function onBeforeLeave(el: Element) {
  const div = el as HTMLDivElement;
  div.style.top = `${div.offsetTop}px`;
  div.style.zIndex = "1";
}
</script>

<template>
  <div class="toast-notification-area">
    <TransitionGroup name="toasts" tag="div" class="toasts" @before-leave="onBeforeLeave">
      <div v-for="toast in toastsStore.toasts" :key="toast.id">
        <ToastNotification
          :id="toast.id"
          :message="toast.message"
          :type="toast.type"
          :count="toast.count"
        />
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
  width: 60dvw;
  padding: var(--spacing-padding-large);

  & .toasts {
    position: relative;
    display: flex;
    flex-direction: column;
    gap: var(--spacing-gap-normal);

    & > div {
      z-index: 2;
    }
  }
}

.toasts-move,
.toasts-enter-active,
.toasts-leave-active {
  transition: all 0.3s ease;
}

.toasts-enter-from {
  opacity: 0;
  transform: translateY(30px);
}

.toasts-leave-to {
  opacity: 0;
}

.toasts-leave-active {
  position: absolute;
  width: 100%;
}
</style>

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
          :dismissible="toast.dismissible"
          :duration="toast.duration"
        />
      </div>
    </TransitionGroup>
  </div>
</template>

<style scoped>
.toast-notification-area {
  position: fixed;
  z-index: 9000;
  top: 0;
  right: 0;
  width: 100dvw;
  padding: var(--spacing-20);
  pointer-events: none;

  & .toasts {
    position: relative;
    display: flex;
    flex-direction: column-reverse;
    align-items: center;
    gap: var(--spacing-12);

    & > div {
      z-index: 2;
      width: 40%;
    }
  }
}

.toasts-move,
.toasts-enter-active,
.toasts-leave-active {
  transition: all var(--animation-duration) var(--animation-easing);
}

.toasts-enter-from {
  opacity: 0;
  transform: translateY(-30px);
}

.toasts-leave-to {
  opacity: 0;
}

.toasts-leave-active {
  position: absolute;
  width: 100%;
}
</style>

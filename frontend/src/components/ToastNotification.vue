<script setup lang="ts">
import { useToastsStore, type Toast } from "@/stores/toasts";
import { computed } from "vue";

const props = defineProps<Toast>();
const toastsStore = useToastsStore();

const icon = computed(() => {
  switch (props.type) {
    case "success":
      return "icon-success";
    case "error":
      return "icon-error";
    case "info":
      return "icon-info";
    case "warning":
      return "icon-warning";
    default:
      return "icon-info";
  }
});

function onDismiss() {
  toastsStore.dismissToast(props.id);
}
</script>

<template>
  <div class="toast" :class="props.type">
    <div class="message">
      <span class="icon" :class="icon"></span>
      {{ props.message }}
      <span class="count" v-if="props.count && props.count > 1">(x{{ props.count }})</span>
    </div>
    <div class="actions">
      <button @click="onDismiss">dismiss</button>
    </div>
  </div>
</template>

<style scoped>
.toast {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: var(--spacing-padding-normal) var(--spacing-padding-large);
  border: 2px solid var(--toast-border-color);
  border-radius: 32px;
  background-color: var(--toast-background-color);
  background-image: linear-gradient(to right, var(--toast-background-color) 20%, transparent 100%),
    repeating-linear-gradient(
      45deg,
      var(--toast-stripe-color),
      var(--toast-stripe-color) 1px,
      var(--toast-background-color) 1px,
      var(--toast-background-color) 15px
    );
  box-shadow:
    0 4px 18.2px -2px var(--toast-shadow-elevation),
    inset 0 0 1.1px 2px var(--toast-shadow-inset);

  & .icon {
    font-size: calc(13px * 1.5);
  }

  &.success .icon {
    color: #77bf85;
  }

  &.error .icon {
    color: #f05454;
  }

  &.info .icon {
    color: #3f72af;
  }

  &.warning .icon {
    color: #f08b30;
  }

  & .message {
    display: flex;
    align-items: center;
    color: var(--toast-text-color);
    font-size: var(--text-size-highlight);
    font-weight: var(--text-weight-highlight);
    gap: var(--spacing-gap-normal);
  }

  & .actions {
    display: flex;
    align-items: center;
    gap: var(--spacing-gap-normal);

    & button {
      border: none;
      background: none;
      color: var(--toast-dismiss-color);
      cursor: pointer;
      font-size: var(--text-size-normal);
      font-weight: var(--text-weight-normal);
    }
  }
}
</style>

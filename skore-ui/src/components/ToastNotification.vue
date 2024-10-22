<script setup lang="ts">
import { useToastsStore, type Toast } from "@/stores/toasts";
import { computed, onBeforeUnmount, onMounted, ref } from "vue";

const props = defineProps<Toast>();
const toastsStore = useToastsStore();
let durationTimer = -1;
let animationPlayState = ref("paused");

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
  cancelDurationTimer();
  toastsStore.dismissToast(props.id);
}

function cancelDurationTimer() {
  if (durationTimer !== -1) {
    clearInterval(durationTimer);
  }
}

onMounted(() => {
  if (props.duration !== undefined && props.duration !== Infinity) {
    durationTimer = window.setInterval(() => {
      onDismiss();
    }, props.duration * 1000);
    animationPlayState.value = "running";
  }
});

onBeforeUnmount(() => {
  cancelDurationTimer();
});
</script>

<template>
  <div class="toast" :class="props.type">
    <div class="message">
      <div class="icon-wrapper">
        <div class="countdown" v-if="props.duration !== undefined && props.duration !== Infinity">
          <div class="background" />
          <div
            class="foreground"
            :style="{
              animationPlayState,
              animationDuration: `${props.duration ?? 0}s`,
            }"
          />
        </div>
        <div class="icon">
          <span :class="icon"></span>
        </div>
      </div>
      {{ props.message }}
      <span class="count" v-if="props.count && props.count > 1">(x{{ props.count }})</span>
    </div>
    <div class="actions" v-if="props.dismissible">
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
    color: #d63232;
  }

  &.info .icon {
    color: #4b44ff;
  }

  &.warning .icon {
    color: #ff9f0a;
  }

  & .message {
    z-index: 2;
    display: flex;
    align-items: center;
    color: var(--toast-text-color);
    font-size: var(--text-size-highlight);
    font-weight: var(--text-weight-highlight);
    gap: var(--spacing-gap-normal);

    & .icon-wrapper {
      & .icon {
        position: relative;
        z-index: 2;
        display: flex;
        width: 20px;
        height: 20px;
      }

      & .countdown {
        position: relative;
        z-index: 1;

        & .background,
        & .foreground {
          position: absolute;
          top: 0;
          left: 0;
          width: 19px;
          height: 19px;
          box-sizing: border-box;
          padding: 3px; /* the boder thickness */
          border-radius: 50%;
          aspect-ratio: 1;
        }

        & .background {
          background-color: #2f3037;
          mask:
            linear-gradient(#0000 0 0) content-box intersect,
            conic-gradient(#000 360deg, #0000 0);
        }

        & .foreground {
          animation-duration: 10s;
          animation-iteration-count: 1;
          animation-name: countdown;
          animation-play-state: paused;
          animation-timing-function: linear;
          background-color: white;
          mask:
            linear-gradient(#0000 0 0) content-box intersect,
            conic-gradient(#000 var(--progress), #0000 0);

          --progress: 0deg; /* control the progression */
        }
      }
    }
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

@property --progress {
  inherits: false;
  initial-value: 0deg;
  syntax: "<angle>";
}

@keyframes countdown {
  from {
    --progress: 0deg;
  }

  to {
    --progress: 360deg;
  }
}
</style>

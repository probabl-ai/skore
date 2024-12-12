<script setup lang="ts">
import { computed, useTemplateRef } from "vue";

const props = defineProps<{
  value: number;
}>();

const container = useTemplateRef<HTMLDivElement>("container");
const knobPosition = computed<string>(() => {
  if (container.value) {
    const width = container.value.clientWidth;
    const left = props.value * width;
    return `${left}px`;
  }
  return "0";
});
</script>

<template>
  <div class="static-range" ref="container">
    <svg viewBox="0 0 100 3" fill="none" preserveAspectRatio="none" class="track">
      <rect width="100" height="3" rx="1.5" fill="#E8EBEF" />
      <rect :width="props.value * 100" height="3" rx="1.5" fill="url(#paint0_linear_4987_735)" />
      <defs>
        <linearGradient
          id="paint0_linear_4987_735"
          x1="0"
          y1="0"
          x2="100"
          y2="0"
          gradientUnits="userSpaceOnUse"
        >
          <stop stop-color="#FFC917" />
          <stop offset="0.789976" stop-color="#8DF4BD" />
        </linearGradient>
      </defs>
    </svg>
    <svg class="knob" viewBox="0 0 12 12" :style="{ '--knob-position': knobPosition }">
      <circle cx="6" cy="6" r="6" fill="#E8EBEF" />
      <circle cx="6" cy="6" r="4" fill="white" />
    </svg>
  </div>
</template>

<style scoped>
.static-range {
  position: relative;
  height: 12px;

  & .track,
  & .knob {
    position: absolute;
    left: 0;
  }

  & .track {
    top: 5px;
    width: 100%;
    height: 3px;
  }

  & .knob {
    top: 0;
    left: var(--knob-position);
    width: 12px;
    height: 12px;
    transition: left var(--animation-duration) var(--animation-easing);
  }
}
</style>

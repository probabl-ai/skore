<script setup lang="ts">
import { autoUpdate, offset, useFloating, type Placement } from "@floating-ui/vue";
import { ref } from "vue";

const props = defineProps<{ text?: string; placement?: Placement }>();

const isHover = ref(false);
const reference = ref<HTMLElement>();
const floating = ref<HTMLDivElement>();
const { floatingStyles } = useFloating(reference, floating, {
  middleware: [offset(10)],
  placement: props.placement ?? "bottom",
  whileElementsMounted: autoUpdate,
});
</script>

<template>
  <span
    class="floating-tooltip"
    ref="reference"
    @mouseenter="isHover = true"
    @mouseleave="isHover = false"
  >
    <slot></slot>
    <Transition name="fade">
      <span
        v-if="isHover"
        ref="floating"
        class="floating-tooltip-content"
        :class="props.placement"
        :style="floatingStyles"
      >
        <template v-if="$slots.tooltipContent">
          <slot name="tooltipContent"></slot>
        </template>
        <template v-else>
          {{ props.text }}
        </template>
      </span>
    </Transition>
  </span>
</template>

<style scoped>
.floating-tooltip {
  position: relative;

  .floating-tooltip-content {
    position: absolute;
    z-index: 9999;
    width: max-content;
    padding: var(--spacing-padding-small);
    border: solid var(--border-width-small) var(--border-color-normal);
    border-radius: var(--border-radius);
    background-color: var(--background-color-normal);
    box-shadow: 0 4px 18.2px -2px var(--toast-shadow-elevation);
  }
}
</style>

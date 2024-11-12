<script lang="ts">
export interface ButtonProps {
  label?: string;
  icon?: string;
  isPrimary?: boolean;
  isInline?: boolean;
}
</script>

<script setup lang="ts">
import { generateRandomMultiple } from "@/services/utils";
import { onBeforeMount, ref } from "vue";

const props = withDefaults(defineProps<ButtonProps>(), { isPrimary: false });
const randomBackgroundOffsetX = ref(0);
const randomBackgroundOffsetY = ref(0);

function randomizeBackground() {
  randomBackgroundOffsetX.value = generateRandomMultiple(3, 0, 300);
  randomBackgroundOffsetY.value = generateRandomMultiple(3, 0, 300);
}

onBeforeMount(() => {
  randomizeBackground();
});
</script>

<template>
  <button
    class="button"
    :class="[props.isPrimary ? 'primary' : 'regular', props.isInline ? 'inline' : '']"
    @mouseover="randomizeBackground"
    :style="`
      --background-offset-x: ${randomBackgroundOffsetX}px;
      --background-offset-y: ${randomBackgroundOffsetY}px;
    `"
  >
    <span v-if="props.icon" :class="props.icon" class="icon"></span>
    <span v-if="props.label" class="label">
      {{ props.label }}
    </span>
  </button>
</template>

<style scoped>
.button {
  display: inline-block;
  flex: none;
  flex-direction: row;
  align-items: center;
  justify-content: center;
  border-radius: var(--radius-xs);
  color: var(--color-text-secondary);
  cursor: pointer;

  &.primary {
    padding: var(--spacing-6) var(--spacing-16);
    border: solid 1px var(--color-background-branding);
    background-color: var(--color-background-branding);
    background-image: linear-gradient(to bottom, transparent, var(--color-background-branding)),
      url("../assets/images/button-background.png");
    background-position:
      var(--background-offset-x) var(--background-offset-y),
      0;
    box-shadow:
      0 2px 6px rgb(0 0 0 / 9%),
      0 2px 4px rgb(0 0 0 / 25%),
      inset 0 1px 4.3px rgb(255 226 199 / 84%);
    color: var(--color-text-button-primary);
  }

  &.regular {
    padding: 0 var(--spacing-4);
    border: var(--stroke-width-md) solid var(--color-stroke-background-primary);
    border-radius: var(--radius-xs);
    margin: 0 auto;
    background: var(--color-background-primary);
  }

  &.inline {
    padding: var(--spacing-4);
    border: none;
    background: none;
    box-shadow: none;
  }

  .icon:has(+ .label) {
    padding-right: var(--spacing-8);
  }
}
</style>

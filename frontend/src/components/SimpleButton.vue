<script setup lang="ts">
import { generateRandomMultiple } from "@/services/utils";
import { onBeforeMount, ref } from "vue";

interface ButtonProps {
  label?: string;
  icon?: string;
  isPrimary?: boolean;
}

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
    :class="props.isPrimary ? 'primary' : 'regular'"
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
  border-radius: var(--border-radius);
  box-shadow:
    0 2px 6px rgb(0 0 0 / 9%),
    0 2px 4px rgb(0 0 0 / 25%),
    inset 0 1px 4.3px rgb(255 226 199 / 84%);
  color: var(--button-color);
  cursor: pointer;

  &.primary {
    padding: var(--spacing-padding-small) var(--spacing-padding-normal);
    border: solid 1px rgb(from var(--button-background-color) h s calc(l + 10%));
    background-color: var(--button-background-color);
    background-image: linear-gradient(to bottom, transparent, var(--button-background-color)),
      url("../assets/images/button-background.png");
    background-position:
      var(--background-offset-x) var(--background-offset-y),
      0;
    font-size: var(--text-size-highlight);
    font-weight: var(--text-weight-highlight);
  }

  &.regular {
    padding: calc(var(--spacing-padding-small) * 0.6) var(--spacing-padding-small);
    border: var(--border-width-normal) solid var(--border-color-elevated);
    border-radius: var(--border-radius);
    margin: 0 auto;
    background: var(--border-color-lower);
    box-shadow: inset 0 0 3.24px 2.4px rgb(var(--text-color-highlight) 0.5);
    color: var(--text-color-normal);
    font-size: var(--text-size-small);
    font-weight: var(--text-weight-normal);
  }

  .icon:has(+ .label) {
    padding-right: var(--spacing-padding-small);
  }
}
</style>

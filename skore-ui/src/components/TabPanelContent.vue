<script setup lang="ts">
import { inject } from "vue";

const props = defineProps<{ name: string; icon?: string }>();
const selectedTabName = inject("selectedTabName");
</script>

<template>
  <div class="tab-content" v-if="props.name === selectedTabName">
    <slot />
  </div>
</template>

<style>
.tab-content {
  width: 100%;
  border: solid var(--stroke-width-md) var(--color-stroke-background-primary);
  border-top: 0;
  border-bottom-left-radius: var(--radius-md);
  border-bottom-right-radius: var(--radius-md);
}
</style>

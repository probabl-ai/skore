<script setup lang="ts">
import FloatingTooltip from "@/components/FloatingTooltip.vue";
import SimpleButton from "@/components/SimpleButton.vue";

const props = defineProps<{
  title: string;
  subtitle?: string;
  icon?: string;
  actionIcon?: string;
  actionTooltip?: string;
}>();

const emit = defineEmits(["action"]);
</script>

<template>
  <div class="header">
    <h1>
      <span v-if="props.icon" class="icon" :class="props.icon"></span>{{ title }}
      <span v-if="props.subtitle" class="subtitle">
        <span class="separator">-</span>
        {{ props.subtitle }}
      </span>
    </h1>
    <div class="action" v-if="props.actionIcon">
      <FloatingTooltip :text="props.actionTooltip" v-if="props.actionTooltip">
        <SimpleButton :icon="props.actionIcon" @click="emit('action')" />
      </FloatingTooltip>
      <SimpleButton v-else :icon="props.actionIcon" @click="emit('action')" />
    </div>
  </div>
</template>

<style scoped>
.header {
  display: flex;
  height: var(--height-header);
  align-items: center;
  justify-content: space-between;
  padding: var(--spacing-12);
  border-bottom: solid var(--stroke-width-md) var(--color-stroke-background-primary);
  background-color: var(--color-background-secondary);

  & h1 {
    color: var(--color-text-primary);
    font-size: var(--font-size-sm);
    font-weight: var(--font-weight-regular);
  }

  & .icon {
    padding-right: 4px;
  }

  & .subtitle {
    font-size: var(--font-size-xs);

    & .separator {
      margin: 0 var(--spacing-4);
    }
  }
}
</style>

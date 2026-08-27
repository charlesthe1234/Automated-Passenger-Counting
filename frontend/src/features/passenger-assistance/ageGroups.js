export const AGE_GROUP_FILTERS = [
  { value: "", label: "Any age group", minAge: "", maxAge: "" },
  { value: "kids", label: "Kids (0–12)", minAge: "0", maxAge: "12" },
  { value: "teenager", label: "Teenager (13–19)", minAge: "13", maxAge: "19" },
  { value: "adult", label: "Adult (20–60)", minAge: "20", maxAge: "60" },
  { value: "senior", label: "Senior (61+)", minAge: "61", maxAge: "" },
];

export function formatAgeGroup(value) {
  if (value === null || value === undefined || value === "") return "Unknown";
  const age = Number(value);
  if (!Number.isFinite(age) || age < 0) return "Unknown";
  if (age <= 12) return "Kids";
  if (age <= 19) return "Teenager";
  if (age <= 60) return "Adult";
  return "Senior";
}

export function selectedAgeGroup(filters) {
  return (
    AGE_GROUP_FILTERS.find(
      (group) => group.minAge === filters.min_age && group.maxAge === filters.max_age,
    )?.value || ""
  );
}

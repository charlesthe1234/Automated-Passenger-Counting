import { CheckIcon } from "@radix-ui/react-icons";
import { Monitor, Moon, Sun } from "lucide-react";
import { useEffect, useMemo, useState } from "react";

import { Button } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";

const themes = [
  { id: "light", label: "Light", icon: Sun },
  { id: "dark", label: "Dark", icon: Moon },
  { id: "system", label: "System", icon: Monitor },
];

function getSystemPreference() {
  if (typeof window === "undefined") return "dark";
  return window.matchMedia("(prefers-color-scheme: light)").matches ? "light" : "dark";
}

function getInitialTheme() {
  if (typeof window === "undefined") return "system";
  return window.localStorage.getItem("cag-theme") || "system";
}

export default function ThemeDropdown() {
  const [theme, setTheme] = useState(getInitialTheme);
  const [systemPreference, setSystemPreference] = useState(getSystemPreference);

  const displayTheme = theme === "system" ? systemPreference : theme;
  const CurrentIcon = useMemo(() => {
    return themes.find((item) => item.id === displayTheme)?.icon || Monitor;
  }, [displayTheme]);

  useEffect(() => {
    const mediaQuery = window.matchMedia("(prefers-color-scheme: light)");
    const handleChange = () => setSystemPreference(getSystemPreference());
    mediaQuery.addEventListener("change", handleChange);
    return () => mediaQuery.removeEventListener("change", handleChange);
  }, []);

  useEffect(() => {
    window.localStorage.setItem("cag-theme", theme);
    document.documentElement.dataset.theme = displayTheme;
  }, [displayTheme, theme]);

  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button size="icon" variant="outline" aria-label="Select theme" title="Select theme">
          <CurrentIcon size={16} strokeWidth={2} aria-hidden="true" />
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end" className="min-w-32">
        {themes.map((item) => {
          const Icon = item.icon;
          const selected = theme === item.id;
          return (
            <DropdownMenuItem key={item.id} onClick={() => setTheme(item.id)}>
              <Icon size={16} strokeWidth={2} className="opacity-60" aria-hidden="true" />
              <span>{item.label}</span>
              {selected ? <CheckIcon className="ml-auto h-4 w-4" /> : null}
            </DropdownMenuItem>
          );
        })}
      </DropdownMenuContent>
    </DropdownMenu>
  );
}
